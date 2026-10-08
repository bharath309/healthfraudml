#!/usr/bin/env python3
"""Protocol T step 3: train GB or RF (arm hyper-parameters, random_state=42) on the resampled as-of training matrix for year t
and variant V, score every kept row of year t, and write the scores into results/asof/scores_<t>.parquet (column p_<model>_<V>).
Usage: asof_train.py T VARIANT MODEL [--budget MINUTES]      MODEL in {gb, rf}
Resumable like the arm's train_model.py: warm-start increments with the estimator pickled in state/."""
import os, sys, time, json, pickle, argparse, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from asof_common import DATA, RES, ARM, N_TREES, write_json, sha256, protocol_hash
ap=argparse.ArgumentParser(); ap.add_argument('t',type=int); ap.add_argument('variant'); ap.add_argument('model'); ap.add_argument('--budget',type=float,default=600)
a=ap.parse_args(); t,V,m=a.t,a.variant,a.model; assert V in ('A','B') and m in ('gb','rf'); RS=42; T0=time.time(); tag=f'{t}_{V}'
STATE=os.path.join(ARM,'state'); os.makedirs(STATE,exist_ok=True); sp=os.path.join(STATE,f'asof_{tag}_{m}.pkl')
d=np.load(os.path.join(DATA,f'asof_resampled_{tag}.npz')); X=d['X'].astype('float64'); y=d['y'].astype('int64')
total=N_TREES[m]; step={'gb':30,'rf':100}[m]
st=pickle.load(open(sp,'rb')) if os.path.exists(sp) else {'n':0,'est':None,'fit_seconds':0.0}
if st['est'] is None:
    st['est']=GradientBoostingClassifier(n_estimators=0,max_depth=6,learning_rate=0.05,subsample=0.8,random_state=RS,warm_start=True) if m=='gb' else \
              RandomForestClassifier(n_estimators=0,max_depth=20,min_samples_leaf=5,class_weight='balanced',random_state=RS,n_jobs=-1,warm_start=True)
est=st['est']; left=lambda: a.budget*60-(time.time()-T0)
while st['n']<total:
    inc=min(step,total-st['n']); est.set_params(n_estimators=st['n']+inc); t1=time.time(); est.fit(X,y); dt=time.time()-t1
    st['n']+=inc; st['fit_seconds']+=dt; pickle.dump(st,open(sp,'wb')); print(f'{m} {tag}: {st["n"]}/{total} estimators (+{inc} in {dt:.0f}s)',flush=True)
    if st['n']<total and left()<dt*1.15+20: print('budget reached; resume later',flush=True); sys.exit(0)
te=np.load(os.path.join(DATA,f'asof_test_{tag}.npz')); p=est.predict_proba(te['X'].astype('float64'))[:,1]
yt=te['y'].astype(int); pred=(p>=0.5).astype(int)
r={'t':t,'variant':V,'model':m,'n_estimators':int(st['n']),'fit_seconds':round(st['fit_seconds'],1),'train_rows':int(len(y)),'train_positives':int(y.sum()),'test_rows':int(len(yt)),'test_positives':int(yt.sum()),
   'f1':float(f1_score(yt,pred)),'precision':float(precision_score(yt,pred,zero_division=0)),'recall':float(recall_score(yt,pred)),'auc_pr':float(average_precision_score(yt,p)),'auc_roc':float(roc_auc_score(yt,p)),
   'protocol_sha256':protocol_hash(),'resampled_sha256':sha256(os.path.join(DATA,f'asof_resampled_{tag}.npz'))}
sp_pq=os.path.join(RES,f'scores_{t}.parquet'); col=f'p_{m}_{V}'
S=pd.read_parquet(sp_pq) if os.path.exists(sp_pq) else pd.DataFrame({'npi':te['npi'],'year':np.full(len(yt),t,dtype='int16'),'is_fraud':te['y'],'is_fraud_3yr':te['y3'],'pay':te['pay'],'ptype':te['ptype']})
assert (S['npi'].values==te['npi']).all(); S[col]=p.astype('float32'); S.to_parquet(sp_pq,index=False)
pv=np.load(os.path.join(DATA,f'asof_prev_{tag}.npz')); pp=est.predict_proba(pv['X'].astype('float64'))[:,1]; pq_prev=os.path.join(RES,f'scores_prev_{t}.parquet')
Pp=pd.read_parquet(pq_prev) if os.path.exists(pq_prev) else pd.DataFrame({'npi':pv['npi'],'year':np.full(len(pp),t-1,dtype='int16'),'ptype':pv['ptype']})
assert (Pp['npi'].values==pv['npi']).all(); Pp[col]=pp.astype('float32'); Pp.to_parquet(pq_prev,index=False)
write_json(r,os.path.join(RES,f'fit_{tag}_{m}.json')); print(json.dumps(r))
