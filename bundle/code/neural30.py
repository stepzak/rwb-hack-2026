"""Нейросеть долей: softmax по 1000 маршрутам внутри каждого горизонта."""
import gc,sys
from pathlib import Path
import os
if os.environ.get('RWB_TORCH_PATH'):
    sys.path.insert(0,os.environ['RWB_TORCH_PATH'])
import numpy as np
import pandas as pd
import torch
from torch import nn
from common30 import FINAL

def history(bank):
    if hasattr(bank,'_neural_history'):return bank._neural_history
    arrays={'level':bank.level}
    for n in ['target_1h']+[f'status_{j}' for j in range(1,7)]:
        a=bank.a[n]
        for win in ([6,12,24,96] if n=='target_1h' else [12,48]):
            arrays[(n,win)]=pd.DataFrame(a).rolling(win,min_periods=1).mean().to_numpy(dtype=np.float32).clip(1)
    bank._neural_history=arrays;return arrays

def features(bank,t,cfg):
    t=np.asarray(t,dtype=int);assert len(t) and t.min()>=336 and t.max()<len(bank.times)
    hist=history(bank);level=bank.level[t].clip(1);y=bank.y;cols=[]
    for lag in [0,1,2,3,4,8,12,24,48,49,50,96,144,336]:cols.append(np.log1p(y[t-lag]/level))
    for win in [6,12,24,96]:cols.append(np.log(hist[('target_1h',win)][t]/level))
    for j in range(1,7):
        a=bank.a[f'status_{j}'];own=hist[(f'status_{j}',48)][t]
        for lag in [0,2,8,48]:cols.append(np.log1p(a[t-lag]/own))
        cols.extend([np.log1p(hist[(f'status_{j}',12)][t]/own),np.log1p(own/level)])
    cols.extend([np.log(level/np.maximum(level.mean(axis=1,keepdims=True),1)),np.log1p(y[t]/level)])
    times=bank.times[t];slot=(times.hour*2+times.minute/30).to_numpy()
    for k in [1,2]:
        for fn in [np.sin,np.cos]:cols.append(np.broadcast_to(fn(2*np.pi*k*slot[:,None]/48),level.shape))
    for w in range(7):cols.append(np.broadcast_to((times.dayofweek==w)[:,None],level.shape))
    bases=[]
    for h in range(1,9):
        if cfg.get('base')=='level':b=level.copy()
        else:
            b=np.zeros_like(level)
            for day,weight in [(1,.6),(2,.25),(7,.15)]:
                tt=t+h-48*day
                shape=(y[tt-1]+2*y[tt]+y[tt+1])/4
                b+=weight*shape/np.maximum(bank.level[t-48*day],1)*level
            b=.8*b+.2*y[t]
        b=np.maximum(b,1);bases.append(np.log(b/b.sum(axis=1,keepdims=True)))
    base=np.stack(bases,axis=-1).astype(np.float32)
    x=np.stack(cols,axis=-1).astype(np.float32)
    return np.nan_to_num(x,nan=0,posinf=8,neginf=-8),base

class Net(nn.Module):
    def __init__(self,features,routes,cfg):
        super().__init__();embed=cfg.get('embed',8);width=cfg.get('width',96)
        self.embedding=nn.Embedding(routes,embed) if embed else None
        layers=[nn.Linear(features+embed,width),nn.SiLU(),nn.Dropout(cfg.get('dropout',.1)),nn.Linear(width,width//2),nn.SiLU(),nn.Linear(width//2,8)]
        self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,base):
        if self.embedding is not None:
            emb=self.embedding(torch.arange(x.shape[1],device=x.device))[None].expand(x.shape[0],-1,-1)
            x=torch.cat([x,emb],dim=-1)
        delta=self.net(x).clamp(-4,4)
        return torch.softmax(base+delta,dim=1)

class Neural:
    def __init__(self,cfg):self.cfg=cfg
    def fit(self,bank,origin,folder):
        cfg=self.cfg;torch.set_num_threads(4);torch.manual_seed(2026);torch.set_float32_matmul_precision('highest')
        self.device='cuda' if torch.cuda.is_available() else 'cpu';end=origin-pd.Timedelta(hours=4)
        slot=bank.times.hour*2+bank.times.minute/30
        use=(np.arange(len(bank.times))>=336)&(bank.times>=end-pd.Timedelta(days=cfg.get('days',56)))&(bank.times+pd.Timedelta(hours=4)<end)&(slot>=19)&(slot<=23)
        t=np.flatnonzero(use);assert len(t)>100 and bank.times[t[-1]+8]<end
        x,base=features(bank,t,cfg)
        self.mean=x.mean(axis=(0,1),keepdims=True);self.std=x.std(axis=(0,1),keepdims=True).clip(.05)
        x=np.clip((x-self.mean)/self.std,-8,8)
        labels=np.stack([bank.y[t+h] for h in range(1,9)],axis=-1).astype(np.float32)
        totals=labels.sum(axis=1,keepdims=True);shares=labels/totals
        age=(origin-bank.times[t]).total_seconds().to_numpy()/86400
        weight=np.exp2(-age/cfg.get('half_life',28)).astype(np.float32)
        w=totals[:,0,:]/totals.mean()*weight[:,None]
        X=torch.from_numpy(x).to(self.device);B=torch.from_numpy(base).to(self.device)
        Y=torch.from_numpy(shares).to(self.device);W=torch.from_numpy(w).to(self.device)
        self.model=Net(x.shape[-1],bank.R,cfg).to(self.device)
        opt=torch.optim.AdamW(self.model.parameters(),lr=cfg.get('lr',.001),weight_decay=cfg.get('decay',.01))
        rng=np.random.default_rng(2026);epochs=cfg.get('epochs',12);batch=cfg.get('batch',4)
        for epoch in range(epochs):
            self.model.train();order=rng.permutation(len(t));running=[]
            for a in range(0,len(order),batch):
                ix=torch.as_tensor(order[a:a+batch],device=self.device)
                pred=self.model(X[ix],B[ix]);diff=pred-Y[ix]
                if cfg.get('loss')=='huber':loss=(torch.sqrt(diff.square()+1e-10)-1e-5).sum(dim=1)
                else:loss=diff.abs().sum(dim=1)
                loss=(loss*W[ix]).mean()
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),1);opt.step()
                running.append(float(loss.detach()))
            if epoch in [0,epochs-1]:print('NEURAL',origin,'epoch',epoch+1,'loss',np.mean(running),flush=True)
        self.model.eval();self.audit={'origin':str(origin),'last_train_label':str(bank.times[t[-1]+8]),'train_rows':len(t)*bank.R,'family':'compositional_mlp','epochs':epochs,'device':self.device,'torch_version':torch.__version__}
        if origin==FINAL:
            torch.save({'config':cfg,'mean':self.mean,'std':self.std,'features':x.shape[-1],'routes':bank.R,'state':{k:v.cpu() for k,v in self.model.state_dict().items()}},folder/'final_neural.pt')
        del X,B,Y,W;gc.collect();torch.cuda.empty_cache();return self
    def predict(self,bank,origins):
        t=bank.indices(origins);x,base=features(bank,t,self.cfg);x=np.clip((x-self.mean)/self.std,-8,8)
        with torch.no_grad():p=self.model(torch.from_numpy(x).to(self.device),torch.from_numpy(base).to(self.device)).cpu().numpy()
        return bank.frame(origins),p.transpose(0,2,1).reshape(-1).astype(float)
