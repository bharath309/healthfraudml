#!/usr/bin/env python3
"""Prospective registration, step 1: ONE Gradient Boosting model on the whole CMS-LEIE arm (all kept provider-years 2013-2024,
primary label from the LEIE of 10 Sep 2026), with the arm's frozen recipe: every positive + 20 negatives per positive drawn
with numpy default_rng(42) ("fold 0" = full data; folds 1-5 used 42+k), SMOTE-ENN(random_state=42), then
GradientBoostingClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42) fitted in
warm-start increments of 30 trees (increments verified identical to a single fit, 3 Oct 2026). Resumable (state/full_gb.pkl).
Outputs: models/full_gb.pkl (never committed), out/full_model_report.json (rows, positives, fit seconds, SHA-256 of the pickle).
Usage: train_full.py [--budget 540] [--smoke]   (--smoke: 20 trees on a 2,000-row subsample, for pipeline tests only)"""
import os, sys, time, json, pickle, hashlib, argparse, numpy as np, pandas as pd, pyarrow.parquet as pq, cmsarm
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTEENN
ap=argparse.ArgumentParser(); ap.add_argument('--budget',type=float,default=540); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); T0=time.time()
RS=42; RATIO=20; STATE=os.path.join(cmsarm.HERE,'state'); MOD=os.path.join(cmsarm.HERE,'models'); os.makedirs(STATE,exist_ok=True); os.makedirs(MOD,exist_ok=True)
name='full_gb_smoke' if a.smoke else 'full_gb'; sp=os.path.join(STATE,name+'.pkl'); mp=os.path.join(MOD,name+'.pkl'); rp=os.path.join(cmsarm.OUT,('full_model_report_smoke' if a.smoke else 'full_model_report')+'.json')
if os.path.exists(mp): print('model exists:',mp); sys.exit(0)
def left(): return a.budget-(time.time()-T0)
st=pickle.load(open(sp,'rb')) if os.path.exists(sp) else {'phase':'sample'}
def save(): pickle.dump(st,open(sp,'wb'))
if st['phase']=='sample':
    parts=[]
    for y in cmsarm.YEARS:
        t=pq.read_table(os.path.join(cmsarm.DATA,f'labels_{y}.parquet'),columns=['Rndrng_NPI','year','is_fraud','drop_post_excl']).to_pandas(); t=t[t['drop_post_excl']==0]
        parts.append(pd.DataFrame({'npi':t['Rndrng_NPI'].astype('int64').values,'year':t['year'].astype('int16').values,'y':t['is_fraud'].astype('int8').values}))
    df=pd.concat(parts,ignore_index=True); del parts
    pos=df[df['y']==1]; neg=df[df['y']==0]; rng=np.random.default_rng(RS); n_neg=min(len(neg),RATIO*len(pos))
    sel=pd.concat([pos,neg.iloc[rng.choice(len(neg),size=n_neg,replace=False)]]).sort_values(['year','npi'])
    if a.smoke: sel=sel.sample(n=2000,random_state=0).sort_values(['year','npi'])
    st.update({'phase':'resample','rows_all':int(len(df)),'positives_all':int(df['y'].sum()),'sampled_rows':int(len(sel)),'sampled_pos':int(sel['y'].sum()),'sampled_neg':int(n_neg)})
    sel[['npi','year','y']].to_parquet(os.path.join(cmsarm.DATA,f'train_{name}.parquet'),index=False); save(); print('sampled',st['sampled_rows'],'rows,',st['sampled_pos'],'positives',flush=True)
if st['phase']=='resample':
    keys=pd.read_parquet(os.path.join(cmsarm.DATA,f'train_{name}.parquet')); feats=cmsarm.features('primary')
    X,y,n,yr=cmsarm._load_rows(keys[['npi','year']],feats,'is_fraud',cmsarm.YEARS); X=cmsarm._scale(X,feats)
    assert (y==keys['y'].values).all()
    t=time.time(); Xr,yr_=SMOTEENN(random_state=RS).fit_resample(X,y)
    np.savez_compressed(os.path.join(cmsarm.DATA,f'resampled_{name}.npz'),X=Xr.astype('float32'),y=yr_.astype('int8'))
    st.update({'phase':'train','resampled_rows':int(len(yr_)),'resampled_pos':int(yr_.sum()),'resample_seconds':round(time.time()-t,1),'n':0,'fit_seconds':0.0,'increments':[]}); save(); print('resampled',st['resampled_rows'],'rows in',st['resample_seconds'],'s',flush=True)
if st['phase']=='train':
    d=np.load(os.path.join(cmsarm.DATA,f'resampled_{name}.npz')); X=d['X'].astype('float64'); y=d['y'].astype('int64')
    total=20 if a.smoke else 500; step=30
    if 'est' not in st: st['est']=GradientBoostingClassifier(n_estimators=0,max_depth=6,learning_rate=0.05,subsample=0.8,random_state=RS,warm_start=True)
    est=st['est']
    while st['n']<total:
        inc=min(step,total-st['n']); est.set_params(n_estimators=st['n']+inc); t=time.time(); est.fit(X,y); dt=time.time()-t
        st['n']+=inc; st['fit_seconds']+=dt; st['increments'].append(round(dt,1)); save(); print(f'{st["n"]}/{total} estimators (+{inc} in {dt:.0f}s)',flush=True)
        if st['n']<total and left()<dt*1.15+20: print('budget reached; resume later',flush=True); sys.exit(0)
    est.set_params(warm_start=False); pickle.dump(est,open(mp,'wb'))
    import sklearn, imblearn
    rep={'model':name,'rows_all':st['rows_all'],'positives_all':st['positives_all'],'sampled_rows':st['sampled_rows'],'sampled_pos':st['sampled_pos'],'sampled_neg':st['sampled_neg'],'resampled_rows':st['resampled_rows'],'resampled_pos':st['resampled_pos'],
         'n_estimators':total,'hyper_parameters':{'max_depth':6,'learning_rate':0.05,'subsample':0.8,'random_state':RS},'fit_seconds':round(st['fit_seconds'],1),'increments':st['increments'],'resample_seconds':st['resample_seconds'],
         'model_sha256':hashlib.sha256(open(mp,'rb').read()).hexdigest(),'versions':{'sklearn':sklearn.__version__,'imblearn':imblearn.__version__,'numpy':np.__version__},'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    json.dump(rep,open(rp,'w'),indent=1); os.remove(sp); print(json.dumps(rep,indent=1))
