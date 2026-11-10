"""Историческая форма дня и изменения от текущей точки; только прошлые дни."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from core_features import DATA,CACHE


def extend_profiles(data,cfg):
    if not cfg.get('shape_features'):return data
    signature=hashlib.sha256(Path(__file__).read_bytes()+pd.util.hash_pandas_object(data[['cutoff','horizon','route_id']],index=False).to_numpy().tobytes()).hexdigest()[:16]
    path=CACHE/f'shape_features_{signature}.parquet'
    if path.exists():return pd.concat([data,pd.read_parquet(path)],axis=1)
    raw=pd.read_parquet(DATA/'train_solo_track.parquet',columns=['timestamp','route_id','target_1h']).sort_values(['timestamp','route_id'])
    times=pd.DatetimeIndex(raw.timestamp.unique());routes=np.sort(raw.route_id.unique());R=len(routes)
    assert len(raw)==len(times)*R and not raw.duplicated(['timestamp','route_id']).any()
    a=raw.target_1h.to_numpy(dtype=np.float32).reshape(len(times),R);del raw
    level=pd.DataFrame(a).rolling(48,min_periods=1).mean().to_numpy(dtype=np.float32)
    origins=pd.DatetimeIndex(data.cutoff.unique()).sort_values();ti=times.get_indexer(origins);O=len(ti)
    scale=np.maximum(level[ti],1);days=np.arange(1,29)
    ix=ti[:,None]-48*days[None,:];valid=ix>=0;safe=np.maximum(ix,0)
    pastlevel=np.maximum(level[safe],1).transpose(0,2,1)
    pastcurrent=a[safe].transpose(0,2,1)
    pasttrend=((a[safe]-a[np.maximum(safe-4,0)])/np.maximum(level[safe],1)).transpose(0,2,1)
    todaytrend=(a[ti]-a[ti-4])/scale
    recent=np.exp2(-days/7).astype(np.float32)[None,None,:]
    gated=np.exp(-np.abs(pasttrend-todaytrend[:,:,None])/.25)*recent
    extras={}
    def add(name,h,value):
        if name not in extras:extras[name]=np.empty((O,8,R),dtype=np.float32)
        extras[name][:,h-1]=np.asarray(value,dtype=np.float32)
    def average(x,w):
        weights=np.broadcast_to(w,x.shape)*np.isfinite(x)
        return np.sum(np.nan_to_num(x)*weights,axis=2)/np.maximum(weights.sum(axis=2),1e-12)
    for h in range(1,9):
        future=a[np.minimum(safe+h,len(a)-1)].transpose(0,2,1).copy()
        future[~np.broadcast_to(valid[:,None,:],future.shape)]=np.nan
        z=future/pastlevel
        change=(future-pastcurrent)/pastlevel
        ratio=np.clip(future/np.maximum(pastcurrent,.1*pastlevel),0,10)
        for window in [7,14,28]:
            mask=(days<=window)[None,None,:]
            add(f'shape_level_median{window}',h,np.nanmedian(z[:,:,:window],axis=2))
            add(f'shape_change_recent{window}',h,average(change,mask*recent)+a[ti]/scale)
            add(f'shape_ratio_recent{window}',h,average(ratio,mask*recent)*a[ti]/scale)
        weekly=(days%7==0)[None,None,:]*np.exp2(-days/14)[None,None,:]
        add('shape_level_weekly',h,average(z,weekly))
        add('shape_level_slope_gated',h,average(z,gated))
    frame=pd.DataFrame({k:v.reshape(-1) for k,v in extras.items()})
    assert len(frame)==len(data)
    expected=origins.repeat(8*R);assert np.array_equal(expected.to_numpy(),data.cutoff.to_numpy())
    frame.to_parquet(path,index=False,compression='zstd');print('SHAPE_FEATURES',frame.shape,str(path),flush=True)
    return pd.concat([data,frame],axis=1)
