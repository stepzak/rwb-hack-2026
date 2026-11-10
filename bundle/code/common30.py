"""Общая панель и фиксированный причинный прогноз сетевого объёма."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

ROOT=Path(os.environ.get('RWB30_ROOT',str(Path(__file__).resolve().parents[1])))
CACHE=Path(os.environ.get('RWB30_CACHE',str(ROOT/'.cache')))
CACHE.mkdir(exist_ok=True)
FOLDS=list(pd.date_range('2025-09-20 10:30','2025-10-25 10:30',freq='7D'))
FINAL=pd.Timestamp('2025-11-01 10:30')

def write(path,value):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)

def metric(d,p):
    y=d.y.to_numpy(dtype=float);p=np.asarray(p,dtype=float)
    assert p.shape==y.shape and np.isfinite(p).all() and (p>=0).all() and y.sum()>0
    e=p-y;daily=pd.DataFrame({'day':d.cutoff.to_numpy(),'y':y,'ae':abs(e),'e':e}).groupby('day').sum()
    return {'WAPE':float(abs(e).sum()/y.sum()),'Rbias':float(e.sum()/y.sum()),
            'score':float((abs(e).sum()+abs(e.sum()))/y.sum()),
            'daily_score':float(((daily.ae+abs(daily.e))/daily.y).mean()),'daily_WAPE':float((daily.ae/daily.y).mean())}

class Bank:
    def __init__(self):
        names=['target_1h']+[f'status_{j}' for j in range(1,7)]
        raw=pd.read_parquet(Path(os.environ.get('RWB30_DATA',str(ROOT/'data')))/'train_solo_track.parquet',columns=['timestamp','route_id',*names]).sort_values(['timestamp','route_id'])
        self.times=pd.DatetimeIndex(raw.timestamp.unique());self.routes=np.sort(raw.route_id.unique());self.R=len(self.routes)
        assert self.R==1000 and len(raw)==len(self.times)*self.R and not raw.duplicated(['timestamp','route_id']).any()
        assert self.times.equals(pd.date_range(self.times[0],self.times[-1],freq='30min')) and raw.notna().all().all()
        self.a={n:raw[n].to_numpy(dtype=np.float32).reshape(-1,self.R) for n in names}
        self.y=self.a['target_1h'];self.total=self.y.sum(axis=1,dtype=float)
        self.level=pd.DataFrame(self.y).rolling(48,min_periods=1).mean().to_numpy(dtype=np.float32).clip(1)
        self.level_total=self.level.sum(axis=1,dtype=float)
        self._flat=None

    def indices(self,times):return self.times.get_indexer(pd.DatetimeIndex(times))

    def frame(self,origins):
        ti=self.indices(origins);assert (ti>=0).all()
        pieces=[]
        for t in ti:
            for h in range(1,9):
                future=t+h
                pieces.append(pd.DataFrame({'cutoff':self.times[t],'timestamp':self.times[t]+pd.Timedelta(minutes=30*h),
                    'route_id':self.routes,'horizon':h,'y':self.y[future].astype(float) if future<len(self.times) else np.full(self.R,np.nan)}))
        return pd.concat(pieces,ignore_index=True)

    def flat(self):
        if self._flat is not None:return self._flat
        candidates=list((ROOT.parent/'feature_cache_round2').glob('features_FULL_*_starts5.parquet'))
        if len(candidates)==1:
            feature_path=candidates[0]
        else:
            import feature_builder30
            feature_builder30.DATA=ROOT/'data';feature_builder30.CACHE=CACHE
            feature_path=feature_builder30.build(5)
        d=pd.read_parquet(feature_path)
        ti=self.indices(d.cutoff);ri=np.searchsorted(self.routes,d.route_id.to_numpy());fi=ti+d.horizon.to_numpy()
        known=fi<len(self.times)
        assert np.array_equal(d.loc[known,'y'].to_numpy(),self.y[fi[known],ri[known]])
        assert d.loc[~known,'y'].isna().all()
        # Собственные относительные уровни статусов, только текущая/прошлая история.
        for j in range(1,7):
            a=self.a[f'status_{j}']
            own=pd.DataFrame(a).rolling(48,min_periods=1).mean().to_numpy(dtype=np.float32).clip(1)
            for lag in [0,2,8,48]:d[f'own_s{j}_lag{lag}']=(a[ti-lag,ri]/own[ti,ri]).astype(np.float32)
            recent=pd.DataFrame(a).rolling(12,min_periods=1).mean().to_numpy(dtype=np.float32)
            d[f'own_s{j}_recent']=(recent[ti,ri]/own[ti,ri]).astype(np.float32)
        self._flat=d;return d

def volume_bank(bank):
    """Одна заранее фиксированная политика session9; это компонент каждой модели."""
    path=CACHE/'network_policy9.csv'
    if path.exists():return pd.read_csv(path,parse_dates=['cutoff'])
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    sys.path.insert(0,str(ROOT/'baseline_session6/code'))
    import core_features
    core_features.DATA=ROOT/'data';core_features.CACHE=CACHE
    from core_volume import network_data
    net=network_data();scale=net.scale.to_numpy(dtype=float)/8
    labels=np.column_stack([pd.Series(bank.total,index=bank.times).reindex(net.timestamp+pd.Timedelta(minutes=30*h)).to_numpy() for h in range(1,9)])
    columns=[c for c in net if c.startswith(('target_1h_','future_profile_','weekday_')) or c.endswith('_ownlevel') or c in ['slot','weekday','clock_sin','clock_cos']]
    result=[];audits=[]
    origins=pd.date_range(FOLDS[0]-pd.Timedelta(days=7),FINAL,freq='D')
    for origin in origins:
        tr=(net.label_end.lt(origin-pd.Timedelta(hours=4))&net.timestamp.ge(origin-pd.Timedelta(days=56))).to_numpy(copy=True)
        tr &= np.isfinite(labels).all(axis=1)
        q=net.timestamp.eq(origin).to_numpy();assert q.sum()==1
        age=(origin-net.loc[tr,'timestamp']).dt.total_seconds().to_numpy()/86400
        clock=abs(net.loc[tr,'slot'].to_numpy()-21);clock=np.minimum(clock,48-clock)
        w=np.exp2(-age/14)*np.exp(-.5*(clock/8)**2)
        model=make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),Ridge(alpha=200))
        with threadpool_limits(limits=4):model.fit(net.loc[tr,columns],labels[tr]/scale[tr,None],ridge__sample_weight=w)
        pred=np.maximum(model.predict(net.loc[q,columns]).reshape(8)*scale[q][0],0)
        last=net.loc[tr,'label_end'].max();assert last<origin-pd.Timedelta(hours=4)
        audits.append({'origin':str(origin),'last_label':str(last),'train_rows':int(tr.sum())})
        result.extend({'cutoff':origin,'horizon':h,'volume':float(pred[h-1])} for h in range(1,9))
        if origin==FINAL:
            folder=ROOT/'network_policy';folder.mkdir(exist_ok=True)
            joblib.dump({'model':model,'columns':columns,'last_label':str(last)},folder/'final.joblib',compress=1)
            write(folder/'audit.json',audits)
    result=pd.DataFrame(result);result.to_csv(path,index=False);return result

def project(d,p,volumes):
    p=np.maximum(np.asarray(p,dtype=float),1e-6).copy()
    lookup=volumes.set_index(['cutoff','horizon']).volume
    for key,ix in d.groupby(['cutoff','horizon']).indices.items():p[ix]*=float(lookup.loc[key])/p[ix].sum()
    return p

def blend_reference(d,p,cfg):
    ref=pd.read_parquet(ROOT/'reference/session6_oos.parquet',columns=['cutoff','horizon','route_id','session_prediction'])
    ref=ref.rename(columns={'session_prediction':'base'})
    joined=d[['cutoff','horizon','route_id']].merge(ref,on=['cutoff','horizon','route_id'],validate='one_to_one',how='left')
    known=(joined.base.notna()&d.cutoff.ge('2025-10-11')).to_numpy();base=joined.base.to_numpy(dtype=float)
    cal=(d.cutoff.between('2025-10-11','2025-10-17 23:59').to_numpy())&known
    shaped=p.copy()
    for _,ix in d.loc[known].groupby(['cutoff','horizon']).groups.items():
        ix=np.asarray(list(ix));shaped[ix]*=base[ix].sum()/shaped[ix].sum()
    cap=cfg.get('shape_cap',.85)
    def shape_objective(w):return metric(d.loc[cal],((1-w)*base+w*shaped)[cal])['daily_WAPE']+.001*w*w
    opt=minimize_scalar(shape_objective,bounds=(0,cap),method='bounded')
    alpha=min([0.,cap,float(opt.x)],key=shape_objective)
    z=(1-alpha)*base+alpha*shaped
    full=z.copy()
    for _,ix in d.loc[known].groupby(['cutoff','horizon']).groups.items():
        ix=np.asarray(list(ix));full[ix]*=p[ix].sum()/full[ix].sum()
    def volume_objective(w):return metric(d.loc[cal],((1-w)*z+w*full)[cal])['daily_score']+.002*w*w
    if cfg.get('preserve_base_volume'):beta=0.
    else:
        opt=minimize_scalar(volume_objective,bounds=(0,.75),method='bounded')
        beta=min([0.,.75,float(opt.x)],key=volume_objective)
    out=p.copy();out[known]=((1-beta)*z+beta*full)[known]
    return out,{'shape_weight':alpha,'volume_weight':beta,'fit_dates':'2025-10-11..17','shape_cap':cap}
