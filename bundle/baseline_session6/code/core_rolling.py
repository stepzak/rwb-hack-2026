"""Rolling-origin с тем же возрастом модели и той же калибровкой, что в final.

Ни одна калибровочная метка не входит в train соответствующей модели.
Суммы часовых меток применяются лишь для конкурсной метрики.
"""
from pathlib import Path
import gc,json
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize_scalar


def rescale(data,cfg):
    mode=cfg.get('normalization','mean48')
    if mode=='mean48':return data
    ratio=data['target_mean_12'].to_numpy(dtype=float) if mode=='mean12' else .5*(1+data['target_mean_4'].to_numpy(dtype=float))
    newscale=np.maximum(data.scale.to_numpy(dtype=float)*ratio,1)
    factor=data.scale.to_numpy(dtype=float)/newscale
    data=data.copy()
    columns=[c for c in data if c.startswith(('target_lag_','target_mean_','target_std_','status_','slot_','long_','seq_')) or c=='profile_norm']
    for c in columns:data[c]=(data[c].to_numpy()*factor).astype(np.float32)
    data['scale']=newscale.astype(np.float32);data['log_current_level']=np.log1p(newscale).astype(np.float32)
    return data


def correct(cal,calp,query,p,mode):
    from core_run import metric
    params={};out=p.copy()
    if mode=='none':return out,params
    if mode=='horizon_multiplicative':
        ratios={}
        for h in range(1,9):
            ix=cal.horizon.eq(h).to_numpy();k=float(cal.loc[ix,'y'].to_numpy(dtype=float).sum()/max(calp[ix].sum(),1e-12))
            out[query.horizon.eq(h).to_numpy()]*=k;ratios[str(h)]=k
        params={'multiplier_by_horizon':ratios}
    elif mode=='multiplicative_daily':
        def objective(k):return metric(cal,np.maximum(k*calp,0))['daily_score']
        opt=minimize_scalar(objective,bounds=(.7,1.3),method='bounded',options={'xatol':1e-7})
        k=float(opt.x);out*=k;params={'multiplier':k,'fit':'scipy mean daily WAPE+abs(Rbias)'}
    elif mode=='additive':
        offset=float(np.mean(cal.y.to_numpy(dtype=float)-calp));out+=offset;params={'offset':offset}
    else:raise ValueError(mode)
    return np.maximum(out,0),params


def fit_rolling(data,cfg,result,final=False,trees=None):
    from core_run import columns_for,target_and_offset,write_json
    from catboost import CatBoostRegressor
    data=rescale(data,cfg)
    daily=data.origin_minute.eq(630)
    if final:starts=[pd.Timestamp(data.cutoff.max())]
    else:starts=list(pd.date_range('2025-10-04 10:30','2025-10-25 10:30',freq='7D'))
    pieces=[];audit=[];counts={}
    for origin in starts:
        lag=cfg.get('calibration_days',0)
        calibration_start=origin-pd.Timedelta(days=lag)
        train_end=calibration_start-pd.Timedelta(hours=4)
        train=data.y.notna()&data.timestamp.lt(train_end)
        if cfg.get('train_days'):train &= data.cutoff.ge(train_end-pd.Timedelta(days=cfg['train_days']))
        cal=daily&data.cutoff.ge(calibration_start)&data.cutoff.lt(origin)&data.y.notna()
        query=daily&data.cutoff.ge(origin)&data.cutoff.lt(origin+pd.Timedelta(days=1 if final else 7))
        assert data.loc[train,'timestamp'].max()<train_end
        if lag:
            assert cal.sum()==lag*8000
            assert data.loc[train,'timestamp'].max()+pd.Timedelta(hours=4)<data.loc[cal,'cutoff'].min()
            assert data.loc[cal,'timestamp'].max()<origin-pd.Timedelta(hours=4)
        assert query.sum()==(8000 if final else 56000)
        folder=result/origin.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
        columns=columns_for(data,cfg)
        drop=cfg.get('drop_features',[])
        columns=[c for c in columns if not any(c.startswith(prefix) for prefix in drop)]
        categorical=[c for c in ['route_id'] if c in columns]
        if cfg.get('categorical_weekday'):categorical.append('dayofweek')
        y,_,_=target_and_offset(data.loc[train],cfg)
        _,offset,scale=target_and_offset(data.loc[query],cfg)
        X=data.loc[train,columns];Q=data.loc[query,columns]
        weight=data.loc[train,'scale'].to_numpy(dtype=float);weight/=weight.mean()
        if cfg.get('half_life'):
            age=(train_end-data.loc[train,'cutoff']).dt.total_seconds().to_numpy()/86400
            weight*=np.exp2(-age/cfg['half_life'])
        if cfg.get('weekday_focus'):
            weight*=np.where(data.loc[train,'dayofweek'].to_numpy()==5,cfg['weekday_focus'],1.)
        if cfg.get('origin_sigma'):
            weight*=np.exp(-.5*((data.loc[train,'origin_minute'].to_numpy()-630)/cfg['origin_sigma'])**2)
        n=int(cfg.get('trees',800))
        model=CatBoostRegressor(iterations=n,depth=cfg.get('depth',7),learning_rate=cfg.get('lr',.05),
            loss_function=cfg.get('loss','MAE'),l2_leaf_reg=cfg.get('l2',30),task_type='GPU',devices='0',
            random_seed=cfg.get('seed',42),thread_count=4,cat_features=categorical,one_hot_max_size=cfg.get('one_hot_max_size',2),allow_writing_files=False,verbose=250)
        model.fit(X,y,sample_weight=weight)
        p=np.maximum(model.predict(Q)*scale+offset,0)
        rawp=p.copy();parameters={}
        if lag:
            _,co,cs=target_and_offset(data.loc[cal],cfg)
            cp=np.maximum(model.predict(data.loc[cal,columns])*cs+co,0)
            p,parameters=correct(data.loc[cal],cp,data.loc[query],p,cfg.get('local_calibration','horizon_multiplicative'))
            cframe=data.loc[cal,['cutoff','timestamp','route_id','horizon','y']].copy();cframe['prediction']=cp;cframe.to_parquet(folder/'calibration_predictions.parquet',index=False)
        frame=data.loc[query,['cutoff','timestamp','route_id','horizon','y','profile28']].copy()
        frame['raw_prediction']=rawp;frame['core_prediction']=p;pieces.append(frame)
        model.save_model(str(folder/'model.cbm'))
        joblib.dump({'model':model,'columns':columns,'config':cfg,'calibration':parameters,'train_end':str(train_end)},folder/'model_bundle.joblib',compress=1)
        counts[str(origin)]=n
        audit.append({'origin':str(origin),'train_label_max':str(data.loc[train,'timestamp'].max()),'calibration_start':str(calibration_start),
                      'calibration_label_max':str(data.loc[cal,'timestamp'].max()) if lag else None,'train_rows':int(train.sum()),
                      'query_rows':int(query.sum()),'calibration':parameters})
        print('ROLLING_FOLD',audit[-1],flush=True)
        del model,X,Q;gc.collect()
    write_json(result/'rolling_audit.json',audit)
    return pd.concat(pieces,ignore_index=True).sort_values(['cutoff','horizon','route_id']).reset_index(drop=True),counts
