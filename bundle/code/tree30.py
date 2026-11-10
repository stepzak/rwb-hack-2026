"""Компактные rolling-origin деревья и compositional labels."""
import gc
import joblib
import numpy as np
import pandas as pd
from common30 import FINAL

class Tree:
    def __init__(self,cfg):self.cfg=cfg
    def columns(self,d):
        excluded={'cutoff','timestamp','scale','profile28','y'}
        cols=[c for c in d if c not in excluded]
        if not self.cfg.get('route',False):cols.remove('route_id')
        mode=self.cfg.get('features','relative')
        if mode=='relative':cols=[c for c in cols if not c.startswith(('status_4_','status_5_','status_6_'))]
        if mode=='target':cols=[c for c in cols if not c.startswith(('status_','own_s','network_status'))]
        return cols
    def fit(self,bank,origin,folder):
        self.structural=None
        if self.cfg.get('augment'):
            from structural30 import Structural
            self.structural=Structural(self.cfg['augment']).fit(bank,origin,folder)
        d=bank.flat();end=origin-pd.Timedelta(hours=4)
        tr=(d.timestamp.lt(end)&d.cutoff.ge(end-pd.Timedelta(days=self.cfg.get('days',56)))).to_numpy(copy=True)
        assert tr.any() and d.loc[tr,'timestamp'].max()<end
        cols=self.columns(d);self.cols=cols
        X=d.loc[tr,cols].copy()
        if self.structural is not None:
            X=pd.concat([X,self.structural.extra_features(bank,d.loc[tr])],axis=1)
            cols=list(X.columns);self.cols=cols
        y=d.loc[tr,'y'].to_numpy(dtype=float)
        scale=d.loc[tr,'scale'].to_numpy(dtype=float)
        fi=bank.indices(d.loc[tr,'timestamp']);total=bank.total[fi]
        share=self.cfg.get('target','ratio')
        base=scale if share=='ratio' else scale/np.maximum(bank.level_total[bank.indices(d.loc[tr,'cutoff'])],1)
        if share=='ratio':label=y/scale;weight=scale
        else:label=y/np.maximum(total,1)/base;weight=np.maximum(total,1)*base
        if share=='log_share':label=np.log1p(label)
        weight=weight/weight.mean()
        if self.cfg.get('half_life'):
            age=(end-d.loc[tr,'cutoff']).dt.total_seconds().to_numpy()/86400
            weight*=np.exp2(-age/self.cfg['half_life'])
        cats=[c for c in ['route_id','dayofweek','horizon'] if c in X]
        from catboost import CatBoostRegressor
        self.model=CatBoostRegressor(iterations=self.cfg.get('trees',600),depth=self.cfg.get('depth',7),
            learning_rate=self.cfg.get('lr',.05),loss_function=self.cfg.get('loss','MAE'),
            l2_leaf_reg=self.cfg.get('l2',40),task_type='GPU',devices='0',thread_count=4,
            random_seed=2026,cat_features=cats,one_hot_max_size=12,allow_writing_files=False,verbose=250)
        self.model.fit(X,label,sample_weight=weight)
        if origin==FINAL:
            self.model.save_model(str(folder/'final.cbm'))
            joblib.dump({'columns':cols,'config':self.cfg},folder/'final_metadata.joblib',compress=1)
        self.audit={'origin':str(origin),'last_train_label':str(d.loc[tr,'timestamp'].max()),'train_rows':int(tr.sum())}
        del X;gc.collect();return self
    def predict(self,bank,origins):
        d=bank.flat();use=d.cutoff.isin(origins).to_numpy()
        q=d.loc[use].sort_values(['cutoff','horizon','route_id'])
        X=q[[c for c in self.cols if c in q]].copy()
        if self.structural is not None:X=pd.concat([X,self.structural.extra_features(bank,q)],axis=1)
        p=self.model.predict(X[self.cols]);target=self.cfg.get('target','ratio')
        if target=='log_share':p=np.expm1(p)
        scale=q.scale.to_numpy(dtype=float)
        if target=='ratio':p*=scale
        else:
            den=bank.level_total[bank.indices(q.cutoff)]
            p*=scale/np.maximum(den,1)
        out=q[['cutoff','timestamp','route_id','horizon','y']].copy().reset_index(drop=True)
        return out,np.maximum(p,0)
