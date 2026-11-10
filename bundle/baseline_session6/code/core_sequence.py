"""Прошлая форма ряда: плотные лаги, частотные коэффициенты и наклоны."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from core_features import DATA
CACHE=Path(__file__).resolve().parent/'.cache'


def extend_sequence(data,cfg):
    if not cfg.get('sequence_features'):return data
    cache=CACHE;cache.mkdir(exist_ok=True)
    signature=hashlib.sha256(Path(__file__).read_bytes()+pd.util.hash_pandas_object(data[['cutoff','route_id','horizon']],index=False).to_numpy().tobytes()).hexdigest()[:16]
    path=cache/f'sequence_{signature}.parquet'
    if path.exists():return pd.concat([data,pd.read_parquet(path)],axis=1)
    names=['target_1h']+[f'status_{i}' for i in range(1,7)]
    raw=pd.read_parquet(DATA/'train_solo_track.parquet',columns=['timestamp','route_id',*names]).sort_values(['timestamp','route_id'])
    assert not raw.duplicated(['timestamp','route_id']).any()
    times=pd.DatetimeIndex(raw.timestamp.unique());routes=np.sort(raw.route_id.unique());R=len(routes)
    assert len(raw)==len(times)*R
    values={n:raw[n].to_numpy(dtype=np.float32).reshape(len(times),R) for n in names};del raw
    origins=pd.DatetimeIndex(data.cutoff.unique()).sort_values();ti=times.get_indexer(origins)
    assert (ti>=336).all()
    oi=origins.get_indexer(data.cutoff);ri=np.searchsorted(routes,data.route_id.to_numpy())
    row=oi*R+ri;den=data.scale.to_numpy(dtype=np.float32);extra={}
    def put(name,a):extra['seq_'+name]=(a.reshape(-1)[row]/den).astype(np.float32)
    for name,a in values.items():
        missing=[9,10,11,16,20,32,40] if name=='target_1h' else [3,5,6,7,9,10,11,16,20,32,40]
        for lag in missing:put(f'{name}_lag{lag}',a[ti-lag])
        for window in [12,48]:
            v=a[ti[:,None]-np.arange(window)[None,:]]
            centered=np.arange(window,dtype=np.float32)-(window-1)/2
            slope=np.einsum('otr,t->or',v,centered)/np.sum(centered**2)
            put(f'{name}_slope{window}',slope)
            for freq in [1,2,3]:
                basis=np.cos(np.pi*(np.arange(window)+.5)*freq/window).astype(np.float32)
                coef=np.einsum('otr,t->or',v,basis)/window
                put(f'{name}_dct{window}_{freq}',coef)
    frame=pd.DataFrame(extra);frame.to_parquet(path,index=False,compression='zstd')
    print('SEQUENCE_FEATURES',frame.shape,str(path),flush=True)
    return pd.concat([data,frame],axis=1)
