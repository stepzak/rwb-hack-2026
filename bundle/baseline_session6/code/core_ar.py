"""Маршрутная авторегрессия на расширенной сетке реальных прошлых стартов."""
from pathlib import Path
import gc,hashlib
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from core_features import DATA


def prepare(cfg,query_only=False):
    names=['target_1h']+[f'status_{i}' for i in range(1,7)]
    raw=pd.read_parquet(DATA/'train_solo_track.parquet',columns=['timestamp','route_id',*names]).sort_values(['timestamp','route_id'])
    times=pd.DatetimeIndex(raw.timestamp.unique());routes=np.sort(raw.route_id.unique());R=len(routes);T=len(times)
    assert len(raw)==T*R and not raw.duplicated(['timestamp','route_id']).any()
    assert times.equals(pd.date_range(times[0],times[-1],freq='30min'))
    values={n:raw[n].to_numpy(dtype=np.float32).reshape(T,R) for n in names};del raw
    minutes=times.hour*60+times.minute
    ti=np.flatnonzero((np.arange(T)>=336)&(minutes>=450)&(minutes<=870))
    ti=np.unique(np.append(ti,T-1))
    if query_only:ti=np.array([T-1])
    target=values['target_1h'];mean48=pd.DataFrame(target).rolling(48,min_periods=1).mean().to_numpy(dtype=np.float32)
    scale=np.maximum(mean48[ti],1);O=len(ti);columns=[];arrays=[]
    def add(name,a):columns.append(name);arrays.append(np.asarray(a,dtype=np.float32))
    for lag in [*range(13),16,24,48,96,336]:add(f'target_lag{lag}',target[ti-lag]/scale)
    for window in [4,12]:
        m=pd.DataFrame(target).rolling(window,min_periods=1).mean().to_numpy(dtype=np.float32)
        add(f'target_mean{window}',m[ti]/scale)
    profiles={}
    for day in [1,7]:
        for h in range(1,9):
            p=target[ti+h-48*day]/scale
            profiles[day,h]=p;add(f'profile_day{day}_h{h}',p)
    if cfg.get('ar_status'):
        for name in names[1:]:
            a=values[name];own=np.maximum(pd.DataFrame(a).rolling(48,min_periods=1).mean().to_numpy(dtype=np.float32),1)
            recent=pd.DataFrame(a).rolling(12,min_periods=1).mean().to_numpy(dtype=np.float32)
            add(f'{name}_current',a[ti]/own[ti]);add(f'{name}_recent',recent[ti]/own[ti]);add(f'{name}_delta2',(a[ti]-a[ti-2])/own[ti])
    slot=minutes[ti].to_numpy()
    for freq in [1,2]:
        for label,f in [('sin',np.sin),('cos',np.cos)]:add(f'clock_{label}{freq}',np.broadcast_to(f(2*np.pi*slot[:,None]*freq/1440),(O,R)))
    for day in range(7):add(f'weekday_{day}',np.broadcast_to((times[ti].dayofweek==day)[:,None],(O,R)))
    X=np.stack(arrays,axis=-1);del arrays;gc.collect()
    Y=np.stack([target[np.minimum(ti+h,T-1)] for h in range(1,9)],axis=-1)/scale[:,:,None]
    for h in range(1,9):Y[ti+h>=T,:,h-1]=np.nan
    offset=np.stack([.5*(profiles[1,h]+profiles[7,h]) for h in range(1,9)],axis=-1)
    return {'x':X,'y':Y,'offset':offset,'scale':scale,'times':times[ti].to_numpy(),'routes':routes,'columns':columns}


def fit_regression(x,y,cfg):
    mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),.05)
    X=np.clip((x-mean)/std,-12,12).transpose(1,0,2).astype(np.float64)
    ym=y.mean(axis=0);Y=(y-ym).transpose(1,0,2).astype(np.float64)
    with threadpool_limits(limits=4):
        gram=X.transpose(0,2,1)@X
        eye=np.arange(gram.shape[-1]);gram[:,eye,eye]+=cfg.get('alpha',50.)
        beta=np.linalg.solve(gram,X.transpose(0,2,1)@Y)
    return {'mean':mean,'std':std,'ym':ym,'beta':beta}


def infer(saved,x,offset):
    X=np.clip((x-saved['mean'])/saved['std'],-12,12)
    return np.einsum('orf,rfh->orh',X,saved['beta'],optimize=True)+saved['ym']+offset


def fit_ar(data,cfg,result,final=False,trees=None):
    from core_run import write_json
    from core_rolling import correct
    bank=prepare(cfg);times=bank['times'];R=len(bank['routes']);pieces=[];audit=[]
    starts=[pd.Timestamp(times[-1])] if final else pd.date_range('2025-10-04 10:30','2025-10-25 10:30',freq='7D')
    clock=pd.DatetimeIndex(times);daily=(clock.hour==10)&(clock.minute==30)
    for origin in starts:
        cal_start=origin-pd.Timedelta(days=7);train_end=cal_start-pd.Timedelta(hours=4)
        train=(times+np.timedelta64(4,'h')<np.datetime64(train_end))&(times>=np.datetime64(train_end)-np.timedelta64(cfg.get('train_days',56),'D'))
        train &= np.isfinite(bank['y']).all(axis=(1,2))
        cal=daily&(times>=np.datetime64(cal_start))&(times<np.datetime64(origin))
        query=daily&(times>=np.datetime64(origin))&(times<np.datetime64(origin)+np.timedelta64(1 if final else 7,'D'))
        assert train.sum()>100 and cal.sum()==7 and query.sum()==(1 if final else 7)
        assert pd.Timestamp(times[train].max())+pd.Timedelta(hours=8)<cal_start
        saved=fit_regression(bank['x'][train],(bank['y']-bank['offset'])[train],cfg)
        def frame_for(mask,p):
            out=data.loc[data.cutoff.isin(times[mask])&data.origin_minute.eq(630),['cutoff','timestamp','route_id','horizon','y','profile28']].copy().reset_index(drop=True)
            flat=p.transpose(0,2,1).reshape(-1);assert len(flat)==len(out)
            return out,flat
        raw=np.maximum(infer(saved,bank['x'][query],bank['offset'][query])*bank['scale'][query,:,None],0)
        cp=np.maximum(infer(saved,bank['x'][cal],bank['offset'][cal])*bank['scale'][cal,:,None],0)
        cf,cp=frame_for(cal,cp);qf,p=frame_for(query,raw)
        p,parameters=correct(cf,cp,qf,p,cfg.get('local_calibration','horizon_multiplicative'))
        qf['core_prediction']=p;pieces.append(qf)
        folder=result/origin.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
        saved.update(config=cfg,columns=bank['columns'],routes=bank['routes'],calibration=parameters)
        joblib.dump(saved,folder/'model_bundle.joblib',compress=1)
        audit.append({'origin':str(origin),'train_origins':int(train.sum()),'train_label_max':str(pd.Timestamp(times[train].max())+pd.Timedelta(hours=4)),'calibration_start':str(cal_start),'calibration':parameters})
        print('AR_FOLD',audit[-1],flush=True)
    write_json(result/'ar_audit.json',audit)
    return pd.concat(pieces,ignore_index=True),0
