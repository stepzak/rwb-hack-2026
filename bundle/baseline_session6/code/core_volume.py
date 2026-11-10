"""Причинный прогноз знаменателя метрики: суммы 8 часовых меток по сети.

Это сумма перекрывающихся target, не уникальный объём за четыре часа.
Модель на каждом origin использует только labels до origin минус 4 часа.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import joblib


def network_data():
    from core_features import CACHE,DATA
    signature=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path=CACHE/f'causal_volume_{signature}.parquet'
    if path.exists():return pd.read_parquet(path)
    names=['target_1h']+[f'status_{j}' for j in range(1,7)]
    raw=pd.read_parquet(DATA/'train_solo_track.parquet',columns=['timestamp',*names])
    for name in names:raw[name]=raw[name].astype(np.float64)
    net=raw.groupby('timestamp',sort=True)[names].sum();del raw
    y=net.target_1h;level=y.rolling(48,min_periods=1).mean().clip(lower=1)
    features={}
    for name in names:
        a=net[name]
        for lag in [0,1,2,4,8,12,24,48,96,336,672]:features[f'{name}_lag{lag}']=a.shift(lag)/level
        for window in [4,12,48,336]:features[f'{name}_mean{window}']=a.rolling(window,min_periods=1).mean()/level
        features[f'{name}_ownlevel']=a/a.rolling(48,min_periods=1).mean().clip(lower=1)
        features[f'{name}_delta4']=(a-a.shift(4))/level
    for day in [1,2,7,14]:
        # Все восемь исторических слотов к текущему origin уже известны.
        profile=sum(y.shift(day*48-h) for h in range(1,9))
        features[f'future_profile_day{day}']=profile/(8*level)
        features[f'future_profile_day{day}_adjusted']=profile/(8*level.shift(day*48))
    times=pd.DatetimeIndex(net.index)
    features['slot']=times.hour*2+times.minute//30
    features['weekday']=times.dayofweek
    for weekday in range(7):features[f'weekday_{weekday}']=(times.dayofweek==weekday).astype(np.int8)
    features['clock_sin']=np.sin(2*np.pi*features['slot']/48)
    features['clock_cos']=np.cos(2*np.pi*features['slot']/48)
    result=pd.DataFrame(features,index=times).reset_index(names='timestamp')
    result['scale']=8*level.to_numpy()
    result['y']=sum(y.shift(-h) for h in range(1,9)).to_numpy()
    result['label_end']=result.timestamp+pd.Timedelta(hours=4)
    result['ratio']=result.y/result.scale
    result=result.iloc[672:].reset_index(drop=True)
    result.to_parquet(path,index=False)
    print('VOLUME_DATA',result.shape,path,flush=True)
    return result


def template_for(frame,cfg,final):
    root=Path(__file__).resolve().parents[2];source=cfg.get('shape_source',10)
    if final:
        if source==10:
            keys=pd.read_parquet(root/'inputs/test_keys.parquet')
            old=pd.read_csv(root/'final/submission_candidate.csv')
            p=frame[['timestamp','route_id']].merge(keys,on=['timestamp','route_id'],validate='one_to_one',how='left').merge(old,on='id',validate='one_to_one',how='left').y_pred.to_numpy()
        elif source==40:
            d=pd.read_parquet(root/'final_round40_iter40/prediction_details.parquet')
            p=frame[['timestamp','route_id']].merge(d[['timestamp','route_id','prediction']],on=['timestamp','route_id'],validate='one_to_one',how='left').prediction.to_numpy()
        else:raise ValueError(source)
    else:
        ref=pd.read_parquet(root/f'results/iteration_{source:02}/predictions.parquet').sort_values(['cutoff','horizon','route_id']).reset_index(drop=True)
        assert np.array_equal(frame[['cutoff','horizon','route_id']].to_numpy(),ref[['cutoff','horizon','route_id']].to_numpy())
        p=ref.calibrated_prediction.to_numpy().copy() if source==10 else ref.prediction.to_numpy().copy()
        if source==10:
            cross=pd.read_parquet(root.parent/'feature_cache_round2/reference_crossfit_4to10.parquet')
            before=frame.cutoff.lt('2025-10-18').to_numpy();p[before]=cross.calibrated_prediction.to_numpy()[before]
        # source40.postprocess k оценены по 11–17: на подборе нужна независимая форма.
        elif source==40:
            use=ref.cutoff.lt('2025-10-11').to_numpy();p=ref.core_prediction.to_numpy(dtype=float).copy()
            for h in range(1,9):
                mask=use&ref.horizon.eq(h).to_numpy();k=ref.loc[mask,'y'].to_numpy(dtype=float).sum()/p[mask].sum()
                p[ref.horizon.eq(h).to_numpy()]*=k
    assert np.isfinite(p).all() and (p>=0).all()
    return p


def fit_volume(data,cfg,result,final=False,trees=None):
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from core_run import write_json
    net=network_data()
    columns=[c for c in net if c not in ['timestamp','label_end','scale','y','ratio']]
    if cfg.get('volume_features')=='target':columns=[c for c in columns if not c.startswith('status_')]
    query=data.cutoff.eq(data.cutoff.max()) if final else data.cutoff.between('2025-10-04','2025-10-31 23:59')
    query &= data.origin_minute.eq(630)
    frame=data.loc[query,['cutoff','timestamp','route_id','horizon','y','profile28']].copy().reset_index(drop=True)
    template=template_for(frame,cfg,final)
    prediction=np.empty(len(frame));audit=[]
    for value in np.sort(frame.cutoff.unique()):
        origin=pd.Timestamp(value)
        train=net.label_end.lt(origin-pd.Timedelta(hours=4))&net.timestamp.ge(origin-pd.Timedelta(days=cfg.get('train_days',42)))&net.ratio.notna()
        q=net.timestamp.eq(origin);assert q.sum()==1 and train.sum()>300
        assert net.loc[train,'label_end'].max()<origin-pd.Timedelta(hours=4)
        age=(origin-net.loc[train,'timestamp']).dt.total_seconds().to_numpy()/86400
        w=np.exp2(-age/cfg.get('half_life',14))
        if cfg.get('clock_sigma'):
            distance=np.abs(net.loc[train,'slot'].to_numpy()-21);distance=np.minimum(distance,48-distance)
            w*=np.exp(-.5*(distance/cfg['clock_sigma'])**2)
        if cfg.get('volume_model','catboost')=='catboost':
            model=CatBoostRegressor(iterations=cfg.get('trees',350),depth=cfg.get('depth',3),learning_rate=cfg.get('lr',.04),
                loss_function=cfg.get('loss','MAE'),l2_leaf_reg=cfg.get('l2',20),cat_features=['weekday'],task_type='CPU',
                thread_count=4,allow_writing_files=False,random_seed=42,verbose=False)
            model.fit(net.loc[train,columns],net.loc[train,'ratio'],sample_weight=w)
        else:
            model=make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),Ridge(alpha=cfg.get('alpha',100)))
            model.fit(net.loc[train,columns],net.loc[train,'ratio'],ridge__sample_weight=w)
        volume=max(0,float(model.predict(net.loc[q,columns])[0]*net.loc[q,'scale'].iloc[0]))
        current=frame.cutoff.eq(origin).to_numpy()
        prediction[current]=template[current]*volume/template[current].sum()
        folder=result/origin.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
        joblib.dump({'model':model,'columns':columns,'config':cfg},folder/'model_bundle.joblib',compress=1)
        audit.append({'origin':str(origin),'train_label_max':str(net.loc[train,'label_end'].max()),'rows':int(train.sum()),'predicted_volume':volume})
    frame['core_prediction']=np.maximum(prediction,0)
    write_json(result/'volume_audit.json',audit)
    print('VOLUME_PREDICTIONS',audit,flush=True)
    return frame,int(cfg.get('trees',0))
