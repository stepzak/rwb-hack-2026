"""Воспроизведение единственного финального CSV из сохранённых весов, без обучения."""
from pathlib import Path
import argparse,importlib.util,json,os,sys
import joblib
import numpy as np
import pandas as pd

class Runtime:
    def __init__(self,bundle,data,cache,neural_device="cpu"):
        self.root=Path(bundle).resolve();self.data=Path(data).resolve();self.cache=Path(cache).resolve()
        self.neural_device=neural_device
        self.cache.mkdir(parents=True,exist_ok=True)
        os.environ['RWB30_ROOT']=str(self.root);os.environ['RWB30_CACHE']=str(self.cache);os.environ['RWB30_DATA']=str(self.data)
        sys.path.insert(0,str(self.root/'code'))
        import common30
        common30.ROOT=self.root;common30.CACHE=self.cache
        self.common=common30;self.bank=common30.Bank();self.raw_cache={}
        spec=importlib.util.spec_from_file_location('baseline6_final_runtime',self.root/'baseline_session6/predict_session.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        base=mod.predict(self.root/'baseline_session6',self.data,self.cache/'base6')
        self.test=pd.read_parquet(self.data/'test_solo_track.parquet')
        assert np.array_equal(base.id,self.test.id)
        self.keys=self.test[['id','timestamp','route_id']].copy()
        self.keys['horizon']=((self.keys.timestamp-common30.FINAL).dt.total_seconds()/1800).astype(int)
        self.keys['base']=base.y_pred.to_numpy(dtype=float)
        self.keys=self.keys.sort_values(['horizon','route_id']).reset_index(drop=True)
        assert set(self.keys.horizon)==set(range(1,9))
        import core_features
        core_features.DATA=self.data;core_features.CACHE=self.cache/'base6'
        from core_volume import network_data
        net=network_data();q=net.timestamp.eq(common30.FINAL)
        saved=joblib.load(self.root/'network_policy/final.joblib')
        v=np.maximum(saved['model'].predict(net.loc[q,saved['columns']]).reshape(8)*float(net.loc[q,'scale'].iloc[0])/8,0)
        self.volumes=pd.DataFrame({'cutoff':common30.FINAL,'horizon':range(1,9),'volume':v})

    def flat(self):
        if self.bank._flat is not None:return self.bank._flat
        # Та же формула признаков, но только финальные 8000 строк.
        import core_features_inference
        core_features_inference.DATA=self.data;core_features_inference.CACHE=self.cache
        d=pd.read_parquet(core_features_inference.build(1))
        b=self.bank;ti=b.indices(d.cutoff);ri=np.searchsorted(b.routes,d.route_id.to_numpy())
        for j in range(1,7):
            a=b.a[f'status_{j}'];own=pd.DataFrame(a).rolling(48,min_periods=1).mean().to_numpy(dtype=np.float32).clip(1)
            for lag in [0,2,8,48]:d[f'own_s{j}_lag{lag}']=(a[ti-lag,ri]/own[ti,ri]).astype(np.float32)
            recent=pd.DataFrame(a).rolling(12,min_periods=1).mean().to_numpy(dtype=np.float32)
            d[f'own_s{j}_recent']=(recent[ti,ri]/own[ti,ri]).astype(np.float32)
        b._flat=d
        assert d.y.isna().all() and len(d)==8000
        return d

    def raw(self,i):
        if i in self.raw_cache:return self.raw_cache[i]
        folder=self.root/'results'/f'iteration_{i:02}';cfg=json.loads((folder/'config.json').read_text())
        kind=cfg['kind'];b=self.bank;c=self.common
        if kind=='tree':
            self.flat()
            from tree30 import Tree
            from catboost import CatBoostRegressor
            model=Tree(cfg);meta=joblib.load(folder/'final_metadata.joblib');model.cols=meta['columns']
            model.model=CatBoostRegressor();model.model.load_model(str(folder/'final.cbm'))
            model.structural=joblib.load(folder/'final_structural.joblib') if cfg.get('augment') else None
            d,p=model.predict(b,[c.FINAL])
        elif kind in ['factor','fourier']:
            model=joblib.load(folder/'final_structural.joblib');d,p=model.predict(b,[c.FINAL])
        elif kind=='neural':
            from neural30 import Neural,Net,torch
            torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
            saved=torch.load(folder/'final_neural.pt',map_location='cpu',weights_only=False)
            model=Neural(cfg);model.mean=saved['mean'];model.std=saved['std'];model.device=self.neural_device
            model.model=Net(saved['features'],saved['routes'],cfg);model.model.load_state_dict(saved['state']);model.model.to(model.device);model.model.eval()
            d,p=model.predict(b,[c.FINAL])
        elif kind=='ensemble':
            recipe=json.loads((folder/'final_ensemble.json').read_text());selected=recipe['selected']
            P=np.column_stack([self.raw(j)[1] for j in selected]);d=self.raw(selected[0])[0].copy()
            p=P@np.array(recipe['weights'])
            for h,w in recipe['by_horizon'].items():
                ix=d.horizon.eq(int(h)).to_numpy();p[ix]=P[ix]@np.array(w)
            for group,w in recipe['regime_weights'].items():
                ix=((d.cutoff.dt.dayofweek>=5)==bool(int(group))).to_numpy();p[ix]=P[ix]@np.array(w)
            for h,k in recipe['volume_multiplier'].items():p[d.horizon.eq(int(h)).to_numpy()]*=k
        else:raise ValueError(kind)
        if not cfg.get('own_volume'):p=c.project(d,p,self.volumes)
        assert np.array_equal(d[['horizon','route_id']].to_numpy(),self.keys[['horizon','route_id']].to_numpy())
        self.raw_cache[i]=(d,np.asarray(p,dtype=float));return self.raw_cache[i]

    def predict(self,i):
        d,p=self.raw(i);folder=self.root/'results'/f'iteration_{i:02}';outer=json.loads((folder/'outer.json').read_text())
        base=self.keys.base.to_numpy(dtype=float);z=np.empty(8000);alpha=outer['shape_weight'];beta=outer['volume_weight']
        for h in range(1,9):
            ix=d.horizon.eq(h).to_numpy();shaped=p[ix]*base[ix].sum()/p[ix].sum()
            route_mix=(1-alpha)*base[ix]+alpha*shaped
            full=route_mix*p[ix].sum()/route_mix.sum()
            z[ix]=(1-beta)*route_mix+beta*full
        details=self.keys[['id']].copy();details['y_pred']=z
        out=self.test[['id']].merge(details,on='id',validate='one_to_one',sort=False)
        assert list(out)==['id','y_pred'] and len(out)==8000 and out.id.is_unique
        assert np.array_equal(out.id,self.test.id) and np.isfinite(out.y_pred).all() and out.y_pred.ge(0).all()
        return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--data',type=Path);parser.add_argument('--cache',type=Path);parser.add_argument('--output',type=Path,default=Path('submission_final.csv'))
    args=parser.parse_args();meta=json.loads((args.bundle/'metadata.json').read_text())
    r=Runtime(args.bundle,args.data or args.bundle/'data',args.cache or args.bundle/'.cache')
    out=r.predict(meta['selected_iteration']);out.to_csv(args.output,index=False);print('Saved',args.output,'rows',len(out),'sum',float(out.y_pred.sum()))
if __name__=='__main__':main()
