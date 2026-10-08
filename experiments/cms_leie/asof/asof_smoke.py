#!/usr/bin/env python3
"""Smoke test for the Protocol T pipeline on a SYNTHETIC arm directory (no CMS/LEIE data; no real model). Builds a fake arm
layout (data/feat_<year>.parquet, data/labels_<year>.parquet, data/LEIE_UPDATED.csv, out/feature_list.json, out/scaler_stats.json,
results/oof_fold*.parquet, cmsarm.py copied from the real arm), then runs asof_labels -> asof_build -> asof_train -> asof_metrics
with ASOF_SMOKE=1 (20 trees) and ASOF_BOOT=50, and checks the outputs' shapes and invariants. Usage: asof_smoke.py WORKDIR"""
import os, sys, json, shutil, subprocess, numpy as np, pandas as pd
W=os.path.abspath(sys.argv[1]); HERE=os.path.dirname(os.path.abspath(__file__)); REAL=os.environ.get('REAL_ARM','/home/claude/cms_leie')
for d in ['data','out','results','asof']: os.makedirs(os.path.join(W,d),exist_ok=True)
shutil.copy(os.path.join(REAL,'cmsarm.py'),os.path.join(W,'cmsarm.py'))
for f in ['asof_common.py','asof_labels.py','asof_build.py','asof_train.py','asof_metrics.py','protocol_T_frozen.md']:
    src=os.path.join(HERE,f) if f!='protocol_T_frozen.md' else os.path.join(HERE,'..','protocol_T_frozen.md')
    if os.path.exists(src): shutil.copy(src,os.path.join(W,'asof',f))
FL=json.load(open(os.path.join(REAL,'out','feature_list.json'))); feats=FL['primary']; json.dump(FL,open(os.path.join(W,'out','feature_list.json'),'w'))
rng=np.random.default_rng(7); NPIS=np.arange(1000000000,1000000000+3000); years=list(range(2013,2025)); n_per=1500
# synthetic LEIE: 60 providers excluded on dates spread 2015–2026 under fraud authorities, 10 under other authorities
excl=rng.choice(NPIS,size=70,replace=False); rows=[]
for i,n in enumerate(excl):
    d=pd.Timestamp('2015-01-01')+pd.Timedelta(days=int(rng.integers(0,4200))); code='1128a1' if i<60 else '1128b14'
    rows.append({'NPI':str(n),'EXCLTYPE':code,'EXCLDATE':d.strftime('%Y%m%d'),'REINDATE':'00000000','WAIVERDATE':'00000000'})
cols=['LASTNAME','FIRSTNAME','MIDNAME','BUSNAME','GENERAL','SPECIALTY','UPIN','NPI','DOB','ADDRESS','CITY','STATE','ZIP','EXCLTYPE','EXCLDATE','REINDATE','WAIVERDATE','WVRSTATE']
L=pd.DataFrame(rows); L=L.reindex(columns=cols,fill_value=''); L.to_csv(os.path.join(W,'data','LEIE_UPDATED.csv'),index=False)
end={r['NPI']:pd.Timestamp(r['EXCLDATE']) for r in rows if r['EXCLTYPE']=='1128a1'}
stats={}
for y in years:
    npi=rng.choice(NPIS,size=n_per,replace=False); X=rng.gamma(2.0,1.0,size=(n_per,48)).astype('float32')
    fraud=np.array([str(n) in end and (end[str(n)].year+(end[str(n)].month>=7))>y for n in npi],dtype='int8')
    drop=np.array([str(n) in end and (end[str(n)].year+(end[str(n)].month>=7))<=y for n in npi],dtype='int8')
    X[fraud==1,:6]*=1.8                                           # a weak signal so rankings are not random
    df=pd.DataFrame(X,columns=feats[:48]); df['block_suppressed']=rng.integers(0,2,n_per); df['bene_demo_n_missing']=rng.integers(0,16,n_per)
    df['Rndrng_Prvdr_Type_code']=rng.integers(0,12,n_per); df['Rndrng_Prvdr_State_Abrvtn_code']=rng.integers(0,50,n_per); df['Rndrng_Prvdr_Ent_Cd_code']=0; df['Rndrng_Prvdr_RUCA_code']=rng.integers(0,10,n_per); df['Rndrng_Prvdr_Mdcr_Prtcptg_Ind_code']=1
    df.insert(0,'Rndrng_NPI',npi.astype('int64')); df.insert(1,'year',y); df.insert(2,'is_fraud',fraud); df.insert(3,'is_fraud_mandatory',fraud); df.insert(4,'is_fraud_5yr',fraud); df.insert(5,'drop_post_excl',drop)
    ey=np.array([ (end[str(n)].year+(end[str(n)].month>=7)) if str(n) in end else np.nan for n in npi])
    df.to_parquet(os.path.join(W,'data',f'feat_{y}.parquet'),index=False)
    pd.DataFrame({'Rndrng_NPI':npi.astype('int64'),'year':y,'Rndrng_Prvdr_Ent_Cd':'I','Rndrng_Prvdr_Type':'X','Rndrng_Prvdr_Cntry':'US','Rndrng_Prvdr_State_Abrvtn':'CA','is_fraud':fraud,'is_fraud_mandatory':fraud,'is_fraud_5yr':fraud,'drop_post_excl':drop,'excl_year':pd.array(ey,dtype='Int64')}).to_parquet(os.path.join(W,'data',f'labels_{y}.parquet'),index=False)
    for c in feats: stats.setdefault(c,{'median':float(df[c].median()),'iqr':float(max(df[c].quantile(.75)-df[c].quantile(.25),1e-9))})
json.dump(stats,open(os.path.join(W,'out','scaler_stats.json'),'w'))
# fake hindsight OOF files: 5 folds, p_gb/p_rf correlated with the label
allrows=pd.concat([pd.read_parquet(os.path.join(W,'data',f'feat_{y}.parquet'),columns=['Rndrng_NPI','year','is_fraud','drop_post_excl']) for y in years]); allrows=allrows[allrows['drop_post_excl']==0]
allrows['npi']=allrows['Rndrng_NPI']; allrows['fold']=rng.integers(1,6,len(allrows))
for k in range(1,6):
    f=allrows[allrows['fold']==k]; p=np.clip(rng.beta(1,20,len(f))+0.3*f['is_fraud'].values,0,1)
    pd.DataFrame({'npi':f['npi'].values,'year':f['year'].values.astype('int16'),'is_fraud':f['is_fraud'].values,'p_gb':p.astype('float32'),'p_rf':(p*0.9).astype('float32')}).to_parquet(os.path.join(W,'results',f'oof_fold{k}.parquet'),index=False)
env=dict(os.environ,ASOF_SMOKE='1',ASOF_BOOT='50',ASOF_ARM_DIR=W,PYTHONPATH=W)
def run(*args):
    r=subprocess.run([sys.executable,os.path.join(W,'asof',args[0])]+list(args[1:]),env=env,cwd=os.path.join(W,'asof'),capture_output=True,text=True)
    if r.returncode!=0: print(r.stdout[-2000:]); print(r.stderr[-3000:]); raise SystemExit(f'{args} failed')
    return r.stdout
print(run('asof_labels.py')[-600:])
for t in [2018,2019,2020,2021,2022]:
    for V in ['A','B']:
        run('asof_build.py',str(t),V)
        for m in ['gb','rf']: run('asof_train.py',str(t),V,m)
    print('year',t,'built and trained',flush=True)
print(run('asof_metrics.py')[-2500:])
S=json.load(open(os.path.join(W,'results','asof','summary.json'))); print('decision rules:',json.dumps(S['decision_rules'],indent=1)[:1500])
# invariants
for t in [2018,2019,2020,2021,2022]:
    sc=pd.read_parquet(os.path.join(W,'results','asof',f'scores_{t}.parquet')); assert set(['p_gb_A','p_rf_A','p_gb_B','p_rf_B'])<=set(sc.columns) and (sc['year']==t).all() and sc['npi'].is_unique
    Y=S['years'][str(t)]; a=Y['asof']['gb_A']['0.01']; assert 0<=a['recall']<=1 and a['rows']==int(np.floor(0.01*len(sc)))
    k=pd.read_parquet(os.path.join(W,'data','asof_known.parquet')); kt=k[(k['year']<=t-1)]; assert (kt[f'known_{t}']<=kt['is_fraud']).all()
print('SMOKE OK')
