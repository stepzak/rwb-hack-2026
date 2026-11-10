"""Train-only динамические факторы и Fourier; общая маршрутная AR-поправка."""
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import randomized_svd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common30 import FINAL

def calendar(times,cfg,center):
    times=pd.DatetimeIndex(times);slot=(times.hour*2+times.minute/30).to_numpy(dtype=float)
    day=(times-pd.Timestamp('2025-07-28')).total_seconds().to_numpy()/86400
    arrays=[np.ones(len(times)),(day-center)/28]
    for k in range(1,cfg.get('daily',3)+1):
        arrays.extend([np.sin(2*np.pi*k*slot/48),np.cos(2*np.pi*k*slot/48)])
    weekly=day
    for k in range(1,cfg.get('weekly',4)+1):
        arrays.extend([np.sin(2*np.pi*k*weekly/7),np.cos(2*np.pi*k*weekly/7)])
    for w in range(7):
        mask=(times.dayofweek==w).astype(float)
        arrays.append(mask)
        if cfg.get('interactions'):
            arrays.extend([mask*np.sin(2*np.pi*slot/48),mask*np.cos(2*np.pi*slot/48)])
    return np.column_stack(arrays)

class Structural:
    def __init__(self,cfg):self.cfg=cfg
    def fit(self,bank,origin,folder):
        cfg=self.cfg;end=origin-pd.Timedelta(hours=4)
        e=int(bank.times.searchsorted(end));a=max(672,e-cfg.get('days',56)*48)
        self.a,self.e=a,e
        self.scale=bank.y[a:e].mean(axis=0,dtype=float).clip(1000)
        normalized=bank.y.astype(float)/self.scale
        if cfg.get('transform')=='log':normalized=np.log1p(normalized)
        self.kind=cfg['kind'];self.origin=origin
        if self.kind=='factor':
            self.mean=normalized[a:e].mean(axis=0)
            with threadpool_limits(limits=4):
                _,_,self.components=randomized_svd(normalized[a:e]-self.mean,n_components=cfg.get('rank',16),n_iter=4,random_state=2026)
                scores=(normalized-self.mean)@self.components.T
            reconstructed=scores@self.components+self.mean
            self.scores=scores
            t=np.arange(a+336,e-8)
            X=self.factor_features(bank,t);Y=np.stack([scores[t+h] for h in range(1,9)],axis=1).reshape(len(t),-1)
            age=(e-t)/48;w=np.exp2(-age/cfg.get('half_life',28))
            self.transition=make_pipeline(StandardScaler(),Ridge(alpha=cfg.get('alpha',100)))
            with threadpool_limits(limits=4):self.transition.fit(X,Y,ridge__sample_weight=w)
        else:
            self.center=float(((bank.times[a:e]-pd.Timestamp('2025-07-28')).total_seconds().to_numpy()/86400).mean())
            X=calendar(bank.times[a:e],cfg,self.center)
            age=(e-np.arange(a,e))/48;w=np.exp2(-age/cfg.get('half_life',21))
            self.seasonal=Ridge(alpha=cfg.get('alpha',30),fit_intercept=False)
            with threadpool_limits(limits=4):self.seasonal.fit(X,normalized[a:e],sample_weight=w)
            reconstructed=self.seasonal.predict(calendar(bank.times,cfg,self.center))
        self.residual=normalized-reconstructed
        self.local_lags=cfg.get('local_lags',[0,1,2,4])
        self.status_scales={}
        if cfg.get('local_status'):
            for j in range(1,7):
                name=f'status_{j}'
                self.status_scales[name]=np.maximum(bank.a[name][a:e].mean(axis=0,dtype=float),1)
        t=np.arange(a+336,e-8)
        self.local=None
        if cfg.get('local',True):
            x=self.local_features(bank,t)
            ym=np.stack([self.residual[t+h] for h in range(1,9)],axis=-1)
            mean=x.mean(axis=0);std=x.std(axis=0).clip(.05)
            xx=np.clip((x-mean)/std,-12,12).transpose(1,0,2)
            yymean=ym.mean(axis=0);yy=(ym-yymean).transpose(1,0,2)
            with threadpool_limits(limits=4):
                gram=xx.transpose(0,2,1)@xx
                diag=np.arange(gram.shape[-1]);gram[:,diag,diag]+=cfg.get('local_alpha',100)
                beta=np.linalg.solve(gram,xx.transpose(0,2,1)@yy)
            self.local={'mean':mean,'std':std,'ymean':yymean,'beta':beta}
        self.audit={'origin':str(origin),'last_train_label':str(bank.times[e-1]),
                    'train_rows':e-a,'family':self.kind,'transform_fit_before':str(end)}
        assert bank.times[e-1]<end
        if origin==FINAL:joblib.dump(self,folder/'final_structural.joblib',compress=1)
        return self

    def factor_features(self,bank,t):
        s=self.scores
        arrays=[s[t-lag] for lag in [0,1,2,4,8,48]]
        arrays.extend(s[t+h-day*48] for day in [1,7] for h in [1,4,8])
        times=bank.times[t]
        slot=(times.hour*2+times.minute/30).to_numpy()
        arrays.extend([np.sin(2*np.pi*slot/48)[:,None],np.cos(2*np.pi*slot/48)[:,None]])
        arrays.extend((times.dayofweek==w).astype(float)[:,None] for w in range(7))
        return np.column_stack(arrays)

    def local_features(self,bank,t):
        arrays=[self.residual[t-lag] for lag in self.local_lags]
        for name,scale in self.status_scales.items():
            arrays.extend([bank.a[name][t]/scale-1,(bank.a[name][t]-bank.a[name][t-2])/scale])
        return np.stack(arrays,axis=-1)

    def predict_tensor(self,bank,t):
        if self.kind=='factor':
            latent=self.transition.predict(self.factor_features(bank,t)).reshape(len(t),8,-1)
            norm=latent@self.components+self.mean
        else:
            future=[bank.times[x]+pd.Timedelta(minutes=30*h) for x in t for h in range(1,9)]
            norm=self.seasonal.predict(calendar(future,self.cfg,self.center)).reshape(len(t),8,bank.R)
        if self.local is not None:
            local=self.local
            x=np.clip((self.local_features(bank,t)-local['mean'])/local['std'],-12,12)
            extra=np.einsum('orf,rfh->orh',x,local['beta'],optimize=True)+local['ymean']
            norm+=extra.transpose(0,2,1)
        if self.cfg.get('transform')=='log':norm=np.expm1(np.clip(norm,-8,5))
        return np.maximum(norm*self.scale,0)

    def predict(self,bank,origins):
        ti=bank.indices(origins);assert (ti<len(bank.times)).all()
        p=self.predict_tensor(bank,ti).reshape(-1)
        return bank.frame(origins),p

    def extra_features(self,bank,part):
        ti=bank.indices(part.cutoff);ri=np.searchsorted(bank.routes,part.route_id.to_numpy())
        scale=part.scale.to_numpy(dtype=float);extra={}
        if self.kind=='factor':
            k=min(self.cfg.get('rank',16),8)
            for j in range(k):
                extra[f'latent_state{j}']=self.scores[ti,j].astype(np.float32)
                extra[f'latent_loading{j}']=self.components[j,ri].astype(np.float32)
                extra[f'latent_product{j}']=(self.scores[ti,j]*self.components[j,ri]*self.scale[ri]/scale).astype(np.float32)
            value=(np.sum(self.scores[ti]*self.components[:,ri].T,axis=1)+self.mean[ri])*self.scale[ri]
        else:
            dates=part.timestamp
            design=calendar(dates,self.cfg,self.center)
            value=np.einsum('np,pn->n',design,self.seasonal.coef_[ri].T)
            value*=self.scale[ri]
        extra['structural_level']=(value/scale).astype(np.float32)
        return pd.DataFrame(extra,index=part.index)
