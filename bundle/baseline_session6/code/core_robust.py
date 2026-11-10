"""Относительные признаки оконных статусов, только из доступной истории.

Отношения описывают сдвиги уровня; физическая вложенность статусов не предполагается.
"""
import numpy as np
import pandas as pd


def extend_robust(data,cfg):
    if not cfg.get('robust_features'):return data
    assert cfg.get('extra_features')=='long'
    new={};floor=1/np.maximum(data.scale.to_numpy(dtype=float),1)
    for j in range(1,7):
        name=f'status_{j}';level=np.maximum(data[f'long_{name}_mean48'].to_numpy(dtype=float),floor)
        week=np.maximum(data[f'long_{name}_mean336'].to_numpy(dtype=float),floor)
        for suffix,key in [('current',f'{name}_lag0'),('mean4',f'{name}_mean4'),('mean12',f'{name}_mean12'),('delta4',f'{name}_delta4')]:
            new[f'robust_s{j}_{suffix}']=np.clip(data[key].to_numpy()/level,-100,100).astype(np.float32)
        new[f'robust_s{j}_level_change']=np.clip(level/week,0,100).astype(np.float32)
        raw=data[f'{name}_lag0'].to_numpy(dtype=float)*data.scale.to_numpy()
        denominator=pd.Series(raw).groupby([data.cutoff,data.horizon],sort=False).transform('sum').to_numpy()
        new[f'robust_s{j}_route_share']=np.divide(raw,np.maximum(denominator,1)).astype(np.float32)
    for window in [4,12]:
        new[f'robust_target_vs{window}']=np.clip(data.target_lag_0.to_numpy()/np.maximum(data[f'target_mean_{window}'].to_numpy(),floor),0,100).astype(np.float32)
    return pd.concat([data,pd.DataFrame(new,index=data.index)],axis=1)
