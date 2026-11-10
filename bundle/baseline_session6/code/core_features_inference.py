"""Причинные панельные признаки; исходные parquet не изменяются."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd

CACHE=Path(__file__).resolve().parent/'.cache'
DATA=Path(__file__).resolve().parent/'data'


def build(starts=1):
    identity=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    path=CACHE/f'features_INFERENCE_{identity}_starts{starts}.parquet'
    if path.exists():return path
    started=time.monotonic()
    names=['target_1h']+[f'status_{j}' for j in range(1,7)]
    d=pd.read_parquet(DATA/'train_solo_track.parquet',columns=['timestamp','route_id',*names]).sort_values(['timestamp','route_id'])
    assert not d.duplicated(['timestamp','route_id']).any() and d.notna().all().all()
    times=pd.DatetimeIndex(d.timestamp.unique());routes=np.sort(d.route_id.unique());R=len(routes)
    assert R==1000 and len(d)==len(times)*R
    assert times.equals(pd.date_range(times[0],times[-1],freq='30min'))
    values={name:d[name].to_numpy(dtype=np.float32).reshape(len(times),R) for name in names};del d
    target=values['target_1h'];T=len(times)
    minutes=times.hour*60+times.minute
    wanted={1:[630],3:[600,630,660],5:[570,600,630,660,690]}[starts]
    origins=np.flatnonzero(np.isin(minutes,wanted)&(np.arange(T)>=336))
    # Все 28 оценочных дат и финальный origin всегда ровно в 10:30.
    origins=np.array([T-1])
    def past(a,index):
        index=np.asarray(index);out=a[np.clip(index,0,T-1)].astype(np.float32,copy=True)
        out[(index<0)|(index>=T)]=np.nan
        return out
    means={}
    for name,a in values.items():
        for window in (4,12,48):
            means[name,window]=pd.DataFrame(a).rolling(window,min_periods=1).mean().to_numpy(dtype=np.float32)
    scale=np.maximum(means['target_1h',48][origins],1)
    common={}
    for lag in (*range(9),12,24,48,96,336):common[f'target_lag_{lag}']=past(target,origins-lag)/scale
    for window in (4,12,48):common[f'target_mean_{window}']=means['target_1h',window][origins]/scale
    for window in (4,12):common[f'target_std_{window}']=pd.DataFrame(target).rolling(window,min_periods=2).std().to_numpy(dtype=np.float32)[origins]/scale
    for name in names[1:]:
        a=values[name]
        for lag in (0,1,2):common[f'{name}_lag{lag}']=past(a,origins-lag)/scale
        for window in (4,12):common[f'{name}_mean{window}']=means[name,window][origins]/scale
        for lag in (2,4):common[f'{name}_delta{lag}']=(a[origins]-past(a,origins-lag))/scale
    common['log_current_level']=np.log1p(scale)
    for name,a in values.items():
        total=a.sum(axis=1,dtype=np.float64)
        netlevel=pd.Series(total).rolling(48,min_periods=1).mean().to_numpy()
        net12=pd.Series(total).rolling(12,min_periods=1).mean().to_numpy()
        den=np.maximum(netlevel[origins],1)
        for suffix,x in [('level',total[origins]/den),('mean12',net12[origins]/den),
                         ('delta2',(total[origins]-total[origins-2])/den)]:
            common[f'network_{name}_{suffix}']=np.broadcast_to(x[:,None],scale.shape).astype(np.float32)
    common['route_network_share']=target[origins]/np.maximum(target[origins].sum(axis=1,keepdims=True),1)
    frames=[]
    for h in range(1,9):
        slot=origins+h;dates=times[origins]+pd.Timedelta(minutes=30*h)
        arrays={name:a.ravel() for name,a in common.items()}
        seasonal=[]
        for day in (1,2,3,7,14,21,28):
            a=past(target,slot-day*48);arrays[f'slot_day{day}']=(a/scale).ravel()
        for day in (1,7):
            for offset in (-2,-1,1,2):arrays[f'slot_day{day}_offset{offset}']=(past(target,slot-day*48+offset)/scale).ravel()
        stack=np.stack([past(target,slot-day*48) for day in range(1,8)])
        arrays['slot_daily7']=(np.nanmean(stack,axis=0)/scale).ravel()
        stack=np.stack([past(target,slot-day*48) for day in (7,14,21,28)])
        with np.errstate(invalid='ignore'):weekly=np.nanmean(stack,axis=0)
        arrays['slot_weekly4']=(weekly/scale).ravel()
        profile_sum=np.zeros_like(scale,dtype=np.float64);count=np.zeros_like(scale)
        for day in range(1,29):
            a=past(target,slot-day*48)*scale/np.maximum(past(means['target_1h',48],origins-day*48),1)
            valid=np.isfinite(a);profile_sum+=np.nan_to_num(a);count+=valid
        profile=np.divide(profile_sum,count,out=np.zeros_like(profile_sum),where=count>0).astype(np.float32)
        arrays['profile_norm']=(profile/scale).ravel()
        arrays['horizon']=np.full(len(origins)*R,h,dtype=np.int8)
        arrays['dayofweek']=np.repeat(dates.dayofweek,R).astype(np.int8)
        arrays['origin_minute']=np.repeat(times[origins].hour*60+times[origins].minute,R).astype(np.int16)
        arrays['future_hour']=np.repeat(dates.hour+dates.minute/60,R).astype(np.float32)
        frame=pd.DataFrame(arrays)
        frame['cutoff']=times[origins].repeat(R);frame['timestamp']=dates.repeat(R)
        frame['route_id']=np.tile(routes,len(origins)).astype(np.int16)
        frame['scale']=scale.ravel();frame['profile28']=profile.ravel()
        frame['y']=past(target,slot).ravel()
        frames.append(frame)
    result=pd.concat(frames,ignore_index=True).sort_values(['cutoff','horizon','route_id']).reset_index(drop=True)
    assert not result.duplicated(['cutoff','horizon','route_id']).any()
    assert result.loc[result.cutoff.eq(times[-1]),'y'].isna().all()
    assert np.array_equal(result.y.notna().to_numpy(),result.timestamp.le(times[-1]).to_numpy())
    path.parent.mkdir(exist_ok=True)
    result.to_parquet(path,index=False,compression='zstd')
    meta={'features':len(common),'columns':len(result.columns),'rows':len(result),'origins':len(origins),
          'starts':starts,'elapsed_seconds':time.monotonic()-started,'source_sha256':identity,
          'first_origin':str(result.cutoff.min()),'last_origin':str(result.cutoff.max())}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('FEATURES',meta,flush=True)
    return path

if __name__=='__main__':
    import sys
    print(build(int(sys.argv[1]) if len(sys.argv)>1 else 1),flush=True)
