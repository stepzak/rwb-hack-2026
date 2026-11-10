"""Восстановление выбранных моделей на CPU, без переобучения."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


def apply_calibration(p,d,mode,params):
    p=np.asarray(p,dtype=float).copy()
    if mode=='horizon_multiplicative':p*=d.horizon.astype(str).map({str(k):v for k,v in params['multiplier_by_horizon'].items()}).to_numpy()
    elif mode=='horizon_additive':p+=d.horizon.astype(str).map({str(k):v for k,v in params['offset_by_horizon'].items()}).to_numpy()
    elif mode in ['multiplicative','multiplicative_daily','daily_optimal_multiplier','ewma_daily7','ewma_daily14']:p*=params['multiplier']
    elif mode in ['additive','ewma7']:p+=params['offset']
    elif mode!='none':raise ValueError(mode)
    return np.maximum(p,0)


def restore(p,d,cfg):
    norm=cfg.get('target','ratio');scale=d.scale.to_numpy(dtype=float)
    if norm=='raw':return np.maximum(p*cfg.get('raw_scale',100000.),0)
    if norm=='log_ratio':p=np.expm1(p)
    if norm=='asinh_residual':p=np.sinh(p)
    out=p*scale
    if norm in ['residual','asinh_residual']:out+=d.profile28.to_numpy()
    elif norm=='delta_current':out+=d.target_lag_0.to_numpy()*scale
    elif norm=='delta_smooth':out+=d.target_mean_4.to_numpy()*scale
    elif norm=='delta_profile_blend':out+=.5*(d.target_lag_0.to_numpy()*scale+d.profile28.to_numpy())
    return np.maximum(out,0)


class Predictor:
    def __init__(self,bundle,data_dir,old_csv,cache):
        self.bundle=Path(bundle);self.data_dir=Path(data_dir);self.cache=Path(cache);self.cache.mkdir(exist_ok=True,parents=True)
        self.test=pd.read_parquet(self.data_dir/'test_solo_track.parquet',columns=['id','timestamp','route_id'])
        self.keys=self.test.merge(pd.read_csv(old_csv),on='id',validate='one_to_one').sort_values(['timestamp','route_id']).reset_index(drop=True)
        self.old=self.keys.y_pred.to_numpy(dtype=float)
        self.graph=json.loads((self.bundle/'graph.json').read_text());self.candidates=json.loads((self.bundle/'candidates.json').read_text())
        self.features={};self.predictions={}
        import importlib.util,sys
        # Загружаем код именно этого комплекта, независимо от предыдущих импортов ядра.
        for name in ['core_features','core_features_inference','core_extra','core_sequence','core_profiles','core_robust','core_rolling','core_ar','core_volume']:
            spec=importlib.util.spec_from_file_location(name,self.bundle/'code'/f'{name}.py')
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        import core_features,core_features_inference,core_extra,core_sequence,core_profiles,core_ar
        for module in [core_features,core_features_inference,core_extra,core_sequence,core_profiles,core_ar]:
            module.DATA=self.data_dir;module.CACHE=self.cache
        self.feature_builder=core_features_inference

    def data(self,cfg):
        signature=json.dumps({k:cfg.get(k) for k in ['extra_features','robust_features','normalization','sequence_features','shape_features']},sort_keys=True)
        if signature not in self.features:
            from core_extra import extend
            from core_sequence import extend_sequence
            from core_profiles import extend_profiles
            from core_robust import extend_robust
            from core_rolling import rescale
            d=pd.read_parquet(self.feature_builder.build(1))
            # Для статистического лага сохраняется train-граница проверочной модели.
            d=extend(d,cfg,final=cfg.get('extra_features')!='aligned')
            d=extend_robust(d,cfg);d=extend_sequence(d,cfg);d=extend_profiles(d,cfg);d=rescale(d,cfg)
            assert np.array_equal(d[['timestamp','route_id']].to_numpy(),self.keys[['timestamp','route_id']].to_numpy())
            self.features[signature]=d
        return self.features[signature]

    def predict_base(self,meta):
        cfg=meta['config'];d=self.data(cfg);folder=self.bundle/meta['folder']
        if meta.get('per_horizon'):
            p=np.empty(len(d))
            for h in range(1,9):
                use=d.horizon.eq(h).to_numpy();part=d.loc[use]
                saved=joblib.load(folder/f'h{h}.joblib')
                p[use]=restore(saved['model'].predict(part[saved['columns']]),part,saved['config'])
            return p
        saved=joblib.load(folder/'model.joblib')
        if cfg['algorithm']=='joint_forest':
            base=d.loc[d.horizon.eq(1)].reset_index(drop=True);X=base.copy()
            for h in range(2,9):
                part=d.loc[d.horizon.eq(h)].reset_index(drop=True)
                for c in ['slot_daily7','slot_weekly4','profile_norm','slot_day1','slot_day7']:X[f'h{h}_{c}']=part[c].to_numpy()
            X=saved['imputer'].transform(X[saved['columns']]).astype(np.float32)
            if cfg.get('tree_median'):p=np.median(np.stack([t.predict(X) for t in saved['model'].estimators_]),axis=0)
            else:p=saved['model'].predict(X)
            p*=base.scale.to_numpy()[:,None]
            if cfg.get('target')=='residual':p+=d.profile28.to_numpy().reshape(8,-1).T
            return np.maximum(p,0).T.reshape(-1)
        return restore(saved['model'].predict(d[saved['columns']]),d,saved['config'])

    def core(self,i):
        i=str(i)
        if i in self.predictions:return self.predictions[i]
        if i=='10':return self.old
        meta=self.graph[i];kind=meta['kind'];cfg=meta['config']
        if kind=='base':p=self.predict_base(meta)
        elif kind in ['raw_ensemble','shape_consensus']:
            if 'volume_weight' in meta:
                shape=self.core(meta['shape_ensemble']);volume=self.core(meta['volume_source']);a=meta['volume_weight']
                p=shape*((1-a)+a*volume.sum()/shape.sum())
            else:
                P=np.column_stack([self.core(j) for j in meta['components']]);p=np.empty(len(self.old))
                if 'weights_by_horizon' in meta:
                    horizons=self.data({}).horizon.to_numpy()
                    for h in range(1,9):
                        ix=horizons==h;C=P[ix].copy()
                        if kind=='shape_consensus':
                            for j in range(C.shape[1]):C[:,j]*=self.old[ix].sum()/max(C[:,j].sum(),1e-12)
                        p[ix]=C@np.asarray(meta['weights_by_horizon'][str(h)])
                else:
                    if kind=='shape_consensus':
                        for j in range(P.shape[1]):P[:,j]*=self.old.sum()/max(P[:,j].sum(),1e-12)
                    p=P@np.asarray(meta['weights'])
        elif kind=='rolling_route':
            d=self.data(cfg);saved=joblib.load(self.bundle/meta['folder']/'model.joblib');p=np.empty(len(d))
            for h,models in saved['models'].items():
                use=d.horizon.eq(h).to_numpy();part=d.loc[use]
                X=models['scaler'].transform(models['imputer'].transform(part[saved['columns']]))
                pred=models['global_model'].predict(X)
                for route,beta in models['corrections'].items():
                    mask=part.route_id.eq(route).to_numpy();pred[mask]+=cfg.get('local_weight',1.)*np.column_stack([np.ones(mask.sum()),X[mask]])@beta
                p[use]=np.maximum(pred*part.scale.to_numpy(),0)
            p=apply_calibration(p,d,cfg.get('local_calibration','horizon_multiplicative'),meta['inner_calibration'])
        elif kind=='rolling_ar':
            from core_ar import prepare,infer
            d=self.data(cfg);bank=prepare(cfg,query_only=True);saved=joblib.load(self.bundle/meta['folder']/'model.joblib')
            p=infer(saved,bank['x'],bank['offset'])*bank['scale'][:,:,None]
            p=np.maximum(p,0).transpose(0,2,1).reshape(-1)
            p=apply_calibration(p,d,cfg.get('local_calibration','horizon_multiplicative'),saved['calibration'])
        elif kind=='causal_volume':
            from core_volume import network_data
            net=network_data();origin=pd.Timestamp('2025-11-01 10:30');q=net.timestamp.eq(origin);assert q.sum()==1
            saved=joblib.load(self.bundle/meta['folder']/'model.joblib')
            total=max(0.,float(saved['model'].predict(net.loc[q,saved['columns']])[0]*net.loc[q,'scale'].iloc[0]))
            p=self.old*total/self.old.sum()
        else:raise ValueError(kind)
        p=np.maximum(p,0);assert len(p)==8000 and np.isfinite(p).all()
        self.predictions[i]=p
        return p

    def submission(self,name='daily'):
        chosen=self.candidates[name];p=self.core(chosen['iteration']);outer=chosen['outer'];d=self.data({})
        corrected=apply_calibration(p,d,outer['mode'],outer['calibration']);p=(1-outer['weight'])*self.old+outer['weight']*corrected
        frame=self.keys[['timestamp','route_id']].copy();frame['y_pred']=p
        out=self.test.merge(frame,on=['timestamp','route_id'],validate='one_to_one',sort=False)[['id','y_pred']]
        assert len(out)==8000 and out.id.is_unique and np.isfinite(out.y_pred).all() and out.y_pred.ge(0).all()
        return out
