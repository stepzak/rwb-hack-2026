"""Смеси истинных rolling OOF: веса каждого фолда обучаются только раньше него."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from threadpoolctl import threadpool_limits
from common30 import ROOT,FINAL,write
KEY=['cutoff','horizon','route_id']
class Ensemble:
    def __init__(self,cfg):self.cfg=cfg
    def fit(self,bank,origin,folder):
        cfg=self.cfg;self.origin=origin
        candidates=list(range(11,34));frames={}
        for i in candidates:
            path=ROOT/f'results/iteration_{i:02}/predictions.parquet'
            validity=path.parent/'validity.json'
            valid=not validity.exists() or json.loads(validity.read_text()).get('valid_for_selection',True)
            if path.exists() and valid:frames[i]=pd.read_parquet(path).sort_values(KEY).reset_index(drop=True)
        assert len(frames)>15
        first=next(iter(frames.values()));self.d=first[['cutoff','timestamp','horizon','route_id','y']].copy()
        for f in frames.values():
            for c in KEY:assert np.array_equal(f[c].to_numpy(),first[c].to_numpy()),c
        end=origin-pd.Timedelta(hours=4)
        train=(self.d.timestamp.lt(end)&self.d.cutoff.ge(origin-pd.Timedelta(days=42))).to_numpy()
        # По одному представителю каждого семейства; выбор лишь на прошлых OOF.
        families=[list(range(11,17))+[25,26,27],list(range(17,21)),list(range(21,25)),list(range(28,34))]
        selected=[]
        for family in families:
            present=[i for i in family if i in frames]
            if train.sum():
                y=self.d.loc[train,'y'].to_numpy(dtype=float)
                selected.append(min(present,key=lambda i:np.abs(frames[i].loc[train,'raw_prediction'].to_numpy()-y).sum()))
            else:selected.append(present[0])
        if cfg.get('mode')!='average':
            selected=list(dict.fromkeys([*selected,12,13,14,25,27]))
        self.selected=selected;self.P=np.column_stack([frames[i].raw_prediction.to_numpy(dtype=float) for i in selected])
        m=len(selected);initial=np.full(m,1/m);weights=initial.copy();self.by_horizon={};self.regime_weights={};self.optimizer_log=[]
        if train.sum() and cfg.get('mode')!='average':
            yy=self.d.loc[train,'y'].to_numpy(dtype=float);pp=self.P[train]
            days=self.d.loc[train,'cutoff'];codes=pd.factorize(days)[0];n=codes.max()+1
            den=np.bincount(codes,weights=yy);row_w=1/den[codes]/n
            def solve(mask,center,penalty):
                mat=pp[mask];y=yy[mask];w=row_w[mask];w=w/w.sum()*len(w)
                denominator=np.sum(y*w)
                def objective(z):
                    e=mat@z-y
                    value=np.sum(abs(e)*w)/denominator+penalty*np.sum((z-center)**2)
                    gradient=mat.T@(np.sign(e)*w)/denominator+2*penalty*(z-center)
                    return value,gradient
                def fun(z):return objective(z)[0]
                with threadpool_limits(limits=4):
                    opt=minimize(objective,center,jac=True,method='SLSQP',bounds=[(0,.75)]*m,constraints=[{'type':'eq','fun':lambda z:z.sum()-1,'jac':lambda z:np.ones_like(z)}],options={'maxiter':100,'ftol':1e-8})
                z=np.asarray(opt.x,dtype=float)
                feasible=np.isfinite(z).all() and abs(z.sum()-1)<1e-5 and (z>=-1e-8).all() and (z<=.75000001).all()
                accepted=bool(opt.success and feasible and fun(z)<=fun(center)+1e-10)
                self.optimizer_log.append({'rows':int(mask.sum()),'success':bool(opt.success),'accepted':accepted,'message':str(opt.message)})
                if not accepted:return center.copy()
                z=z.clip(0);return z/z.sum()
            weights=solve(np.ones(len(yy),dtype=bool),initial,.002)
            if cfg.get('mode')=='horizon':
                hs=self.d.loc[train,'horizon'].to_numpy()
                for h in range(1,9):self.by_horizon[h]=solve(hs==h,weights,.005)
            if cfg.get('mode')=='weekday':
                wd=self.d.loc[train,'cutoff'].dt.dayofweek.to_numpy()
                for group in [0,1]:
                    mask=(wd>=5)==bool(group)
                    if mask.sum()>=24000:self.regime_weights[group]=solve(mask,weights,.008)
        self.weights=weights
        self.volume_multiplier={h:1. for h in range(1,9)}
        if train.sum() and cfg.get('volume'):
            # Калибровка сетевого объёма по прошедшим OOF; новые labels не используются.
            raw=self.P[train]@weights
            a=self.d.loc[train,['cutoff','horizon','y']].copy();a['p']=raw
            agg=a.groupby(['cutoff','horizon'],as_index=False)[['y','p']].sum()
            for h in range(1,9):
                z=agg.loc[agg.horizon.eq(h)].copy();age=(origin-z.cutoff).dt.total_seconds().to_numpy()/86400
                w=np.exp2(-age/(14 if cfg['volume']=='recent' else 28))
                if cfg['volume']=='weekday':w*=np.where(z.cutoff.dt.dayofweek.to_numpy()==origin.dayofweek,3.,1.)
                yy=z.y.to_numpy();p=z.p.to_numpy()
                def f(k):return np.sum(w*abs(k*p-yy)/yy)/w.sum()+.5*(k-1)**2
                opt=minimize_scalar(f,bounds=(.96,1.04),method='bounded')
                k=min([.96,1.,1.04,float(opt.x)],key=f)
                self.volume_multiplier[h]=1+cfg.get('volume_shrink',.5)*(k-1)
        last=self.d.loc[train,'timestamp'].max() if train.sum() else None
        self.audit={'origin':str(origin),'last_train_label':str(last),'train_rows':int(train.sum()),'family':'rolling_OOF_ensemble','selected':selected,'weights':weights.tolist()}
        if last is not None:assert last<end
        if origin==FINAL:
            write(folder/'final_ensemble.json',{'config':cfg,'selected':selected,'weights':weights.tolist(),
                    'by_horizon':{str(k):v.tolist() for k,v in self.by_horizon.items()},
                    'regime_weights':{str(k):v.tolist() for k,v in self.regime_weights.items()},'volume_multiplier':self.volume_multiplier,'optimizer_log':self.optimizer_log,'audit':self.audit})
        return self
    def predict(self,bank,origins):
        use=self.d.cutoff.isin(origins).to_numpy();d=self.d.loc[use].reset_index(drop=True);P=self.P[use]
        p=P@self.weights
        for h,w in self.by_horizon.items():
            ix=d.horizon.eq(h).to_numpy();p[ix]=P[ix]@w
        for group,w in self.regime_weights.items():
            ix=((d.cutoff.dt.dayofweek>=5)==bool(group)).to_numpy();p[ix]=P[ix]@w
        for h,k in self.volume_multiplier.items():p[d.horizon.eq(h).to_numpy()]*=k
        return d,p
