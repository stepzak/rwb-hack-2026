"""Длинная история и статистический лаг, без физической интерпретации статусов."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from core_features import CACHE,DATA


def extend(data,cfg,final=False):
    variant=cfg.get('extra_features')
    if not variant:return data
    signature=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:10]
    keys=hashlib.sha256(pd.util.hash_pandas_object(data[['cutoff','horizon','route_id']],index=False).to_numpy().tobytes()).hexdigest()[:12]
    identity=f'{variant}_{cfg.get("starts",1)}_{int(final)}_{signature}_{keys}'
    path=CACHE/f'extra_{identity}.parquet'
    if path.exists():
        extra=pd.read_parquet(path);assert len(extra)==len(data)
        return pd.concat([data,extra],axis=1)
    names=['target_1h']+[f'status_{i}' for i in range(1,7)]
    raw=pd.read_parquet(DATA/'train_solo_track.parquet',columns=['timestamp','route_id',*names]).sort_values(['timestamp','route_id'])
    times=pd.DatetimeIndex(raw.timestamp.unique());routes=np.sort(raw.route_id.unique());R=len(routes);T=len(times)
    assert len(raw)==T*R and not raw.duplicated(['timestamp','route_id']).any()
    values={n:raw[n].to_numpy(dtype=np.float32).reshape(T,R) for n in names};del raw
    ti=times.get_indexer(data.cutoff);ri=np.searchsorted(routes,data.route_id.to_numpy());scale=data.scale.to_numpy()
    assert (ti>=0).all()
    extras={}
    for name in names[1:]:
        a=values[name]
        for lag in [4,8,12,24,48,96]:
            ix=ti-lag;extras[f'long_{name}_lag{lag}']=np.where(ix>=0,a[np.maximum(ix,0),ri]/scale,np.nan).astype(np.float32)
        for window in [48,336]:
            mean=pd.DataFrame(a).rolling(window,min_periods=1).mean().to_numpy(dtype=np.float32)
            extras[f'long_{name}_mean{window}']=(mean[ti,ri]/scale).astype(np.float32)
    if variant=='aligned':
        boundary=times[-1] if final else pd.Timestamp('2025-10-04')-pd.Timedelta(hours=4)
        target=values['target_1h'];den=pd.DataFrame(target).rolling(48,min_periods=1).mean().to_numpy(dtype=np.float32)
        den=np.maximum(den,1)
        ix=np.flatnonzero((times<boundary)&(np.arange(T)>=336))
        yy=target[ix]/den[ix];yy-=yy.mean(axis=0)
        yss=np.sum(yy*yy,axis=0)
        a=values['status_6'];diff=np.diff(a,axis=0,prepend=a[:1])/den
        best=np.full(R,-np.inf);lags=np.ones(R,dtype=np.int16)
        for lag in range(1,49):
            x=diff[ix-lag].copy();x-=x.mean(axis=0)
            corr=np.sum(x*yy,axis=0)/np.maximum(np.sqrt(np.sum(x*x,axis=0)*yss),1e-12)
            improve=corr>best;best[improve]=corr[improve];lags[improve]=lag
        h=data.horizon.to_numpy();source=ti+h-lags[ri]
        available=(source<=ti)&(source>=0)&(best[ri]>.05)
        for label,a in [('level',values['status_6']),('delta',np.diff(values['status_6'],axis=0,prepend=values['status_6'][:1]))]:
            extras[f'aligned_s6_{label}']=np.where(available,a[np.clip(source,0,T-1),ri]/scale,np.nan).astype(np.float32)
        extras['aligned_statistical_lag']=lags[ri].astype(np.float32);extras['aligned_train_corr']=best[ri].astype(np.float32)
        (CACHE/f'extra_{identity}_lags.json').write_text(json.dumps({'fit_before':str(boundary),'lags':dict(zip(map(str,routes),map(int,lags)))},indent=2))
    extra=pd.DataFrame(extras);extra.to_parquet(path,index=False,compression='zstd')
    print('EXTRA_FEATURES',variant,len(extra.columns),str(path),flush=True)
    return pd.concat([data,extra],axis=1)
