#!/usr/bin/env python3
"""Prospective registration, step 2: score every CY2024 provider-year that is NOT in the LEIE of 10 Sep 2026 (any authority,
any date, any NPI match) with the full-data Gradient Boosting model; rank by (probability desc, NPI asc); form the three
pre-specified lists — top 1% (floor(0.01 n)), top 5% (floor(0.05 n)) and top 1,000 — and write the commitment file.
The NPI lists stay on disk (prospective/cy2024_lists.json, never committed, never published); what is registered is
prospective/commitment_cy2024.json: population size, list sizes, SHA-256 of each sorted NPI list (newline-joined, no
trailing newline), SHA-256 of the model pickle and of the scores file, the input manifest, and the pre-specified tests.
Usage: score_cy2024.py [--model models/full_gb.pkl] [--smoke]"""
import os, re, json, time, pickle, hashlib, argparse, numpy as np, pandas as pd, pyarrow as pa, pyarrow.csv as pacsv, pyarrow.parquet as pq, cmsarm
ap=argparse.ArgumentParser(); ap.add_argument('--model',default=None); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
name='full_gb_smoke' if a.smoke else 'full_gb'; mp=a.model or os.path.join(cmsarm.HERE,'models',name+'.pkl'); PRO=os.path.join(cmsarm.HERE,'prospective'); os.makedirs(PRO,exist_ok=True)
def sha(path): return hashlib.sha256(open(path,'rb').read()).hexdigest()
def sha_list(npis): return hashlib.sha256('\n'.join(str(int(x)) for x in sorted(npis)).encode()).hexdigest()
# population: CY2024 kept rows minus any NPI present in the LEIE file (any authority)
cols=['LASTNAME','FIRSTNAME','MIDNAME','BUSNAME','GENERAL','SPECIALTY','UPIN','NPI','DOB','ADDRESS','CITY','STATE','ZIP','EXCLTYPE','EXCLDATE','REINDATE','WAIVERDATE','WVRSTATE']
leie_path=os.path.join(cmsarm.DATA,'LEIE_UPDATED.csv')
leie=pacsv.read_csv(leie_path,convert_options=pacsv.ConvertOptions(column_types={c:pa.string() for c in cols})).to_pandas()
npi=leie['NPI'].fillna('').str.strip(); leie_npis=set(npi[npi.str.fullmatch(r'\d{10}')&(npi!='0000000000')].astype('int64'))
lab=pq.read_table(os.path.join(cmsarm.DATA,'labels_2024.parquet'),columns=['Rndrng_NPI','year','is_fraud','drop_post_excl']).to_pandas(); lab=lab[lab['drop_post_excl']==0]
lab['npi']=lab['Rndrng_NPI'].astype('int64'); n_kept=len(lab); in_leie=lab['npi'].isin(leie_npis); pop=lab[~in_leie].copy()
assert (pop['is_fraud']==0).all(), 'a provider in the LEIE survived the exclusion filter'
keys=pop[['npi','year']].copy(); keys['year']=keys['year'].astype('int16')
feats=cmsarm.features('primary'); X,y,n,yr=cmsarm._load_rows(keys,feats,'is_fraud',[2024]); X=cmsarm._scale(X,feats); assert (n==keys['npi'].values).all()
est=pickle.load(open(mp,'rb')); t=time.time(); p=np.concatenate([est.predict_proba(X[s:s+250000])[:,1] for s in range(0,len(X),250000)]); dt=time.time()-t
sc=pd.DataFrame({'npi':n.astype('int64'),'p':p.astype('float64')}).sort_values(['p','npi'],ascending=[False,True],kind='stable').reset_index(drop=True); sc['rank']=np.arange(1,len(sc)+1)
N=len(sc); K={'top1pct':int(np.floor(0.01*N)),'top5pct':int(np.floor(0.05*N)),'top1000':1000}
lists={k:sc['npi'].values[:v].tolist() for k,v in K.items()}
scores_path=os.path.join(PRO,f'cy2024_scores_{name}.parquet'); sc.to_parquet(scores_path,index=False)
json.dump({'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'model':name,'lists':{k:[int(x) for x in v] for k,v in lists.items()}},open(os.path.join(PRO,f'cy2024_lists_{name}.json'),'w'))
man=json.load(open(os.path.join(cmsarm.OUT,'download_manifest.json')))
com={'registration':'Prospective test of CMS-LEIE arm flags against future LEIE releases (Paper 2 / Paper T; drafted 5 Oct 2026)',
     'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'model':{'name':name,'file':os.path.relpath(mp,cmsarm.HERE),'sha256':sha(mp),'report':json.load(open(os.path.join(cmsarm.OUT,('full_model_report_smoke' if a.smoke else 'full_model_report')+'.json')))},
     'population':{'cy2024_kept_rows':int(n_kept),'removed_in_leie_any_authority':int(in_leie.sum()),'scored':int(N),'leie_file':'LEIE_UPDATED.csv of 10 Sep 2026','leie_sha256':sha(leie_path)},
     'ranking':'probability descending, NPI ascending','lists':{k:{'size':int(len(v)),'sha256_sorted_npis':sha_list(v),'min_probability':float(sc['p'].values[len(v)-1])} for k,v in lists.items()},
     'scores_file_sha256':sha(scores_path),'features':{'set':'primary','n':len(feats),'feature_list_sha256':sha(os.path.join(cmsarm.OUT,'feature_list.json')),'scaler_stats_sha256':sha(os.path.join(cmsarm.OUT,'scaler_stats.json'))},
     'input_manifest':man,'scoring_seconds':round(dt,1),
     'outcome':'an NPI counts as a hit when a LEIE release dated after 10 Sep 2026 lists it with an exclusion under 1128(a)(1), (a)(2), (a)(3), (b)(4) or (b)(7) and an EXCLDATE later than 10 Sep 2026',
     'tests':{'checks':'LEIE releases of January, April and July 2027; the July 2027 release is the confirmatory check, earlier counts are descriptive',
              'null':'hits in a list are distributed as a simple random sample of the population: X_k ~ Hypergeometric(N = scored population, K = N_new newly excluded NPIs in the population, n = list size)',
              'alternative':'one-sided, more hits than chance; alpha = 0.01 per list; lists tested: top1pct and top5pct; top1000 is descriptive',
              'predictions':{'top1pct':'lift >= 10 (cross-validated per-year lift 16.4 discounted for hindsight optimism)','top5pct':'lift >= 5 (cross-validated 6.9)','top1000':'descriptive; cross-validated per-year lift 37'},
              'power_note':'with N_new = 47 (nine months at the 2024 rate) the critical counts at alpha 0.01 are 3 (top1pct; power 0.85 at lift 10) and 7 (top5pct; power 0.95 at lift 5)',
              'report':'counts and lifts only; no provider is ever named; a failed prediction is reported as failed'}}
json.dump(com,open(os.path.join(PRO,f'commitment_cy2024_{name}.json'),'w'),indent=1)
print(json.dumps({k:com[k] for k in ['population','lists','scores_file_sha256','scoring_seconds']},indent=1)); print('model sha256',com['model']['sha256'])
