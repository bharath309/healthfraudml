#!/usr/bin/env python3
"""Prospective registration, step 3 (run in 2027): count newly excluded providers in the registered lists and apply the
pre-specified tests. Inputs: a NEW LEIE UPDATED.csv (downloaded only after the registration date), the saved lists
(prospective/cy2024_lists_full_gb.json), the commitment file and the scored population. Outcome definition (frozen):
an NPI in the scored population is a hit when the new file lists it with EXCLTYPE in {1128a1, 1128a2, 1128a3, 1128b4, 1128b7}
and EXCLDATE later than 10 Sep 2026. Test: one-sided hypergeometric, alpha 0.01, for top1pct and top5pct; top1000 descriptive.
Usage: check_prospective.py NEW_LEIE_CSV [--name full_gb] [--confirmatory]"""
import os, re, json, time, hashlib, argparse, numpy as np, pandas as pd, pyarrow as pa, pyarrow.csv as pacsv, cmsarm
from scipy.stats import hypergeom
ap=argparse.ArgumentParser(); ap.add_argument('leie'); ap.add_argument('--name',default='full_gb'); ap.add_argument('--confirmatory',action='store_true'); a=ap.parse_args()
PRO=os.path.join(cmsarm.HERE,'prospective'); FRAUD={'1128a1','1128a2','1128a3','1128b4','1128b7'}; CUT=pd.Timestamp('2026-09-10')
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower()) if s is not None else ''
cols=['LASTNAME','FIRSTNAME','MIDNAME','BUSNAME','GENERAL','SPECIALTY','UPIN','NPI','DOB','ADDRESS','CITY','STATE','ZIP','EXCLTYPE','EXCLDATE','REINDATE','WAIVERDATE','WVRSTATE']
leie=pacsv.read_csv(a.leie,convert_options=pacsv.ConvertOptions(column_types={c:pa.string() for c in cols})).to_pandas()
npi=leie['NPI'].fillna('').str.strip(); ok=npi.str.fullmatch(r'\d{10}')&(npi!='0000000000')
dt=pd.to_datetime(leie['EXCLDATE'].fillna('').str.strip(),format='%Y%m%d',errors='coerce')
new=leie[ok&leie['EXCLTYPE'].map(norm).isin(FRAUD)&(dt>CUT)]; new_npis=set(new['NPI'].astype('int64'))
sc=pd.read_parquet(os.path.join(PRO,f'cy2024_scores_{a.name}.parquet')); pop=set(sc['npi'].astype('int64')); N=len(pop)
K=len(pop&new_npis)
lists=json.load(open(os.path.join(PRO,f'cy2024_lists_{a.name}.json')))['lists']; com=json.load(open(os.path.join(PRO,f'commitment_cy2024_{a.name}.json')))
rep={'leie_file':os.path.basename(a.leie),'leie_sha256':hashlib.sha256(open(a.leie,'rb').read()).hexdigest(),'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'confirmatory':a.confirmatory,
     'population':N,'new_fraud_authority_exclusions_in_population':K,'lists':{}}
for k,v in lists.items():
    s=set(int(x) for x in v); n=len(s); assert com['lists'][k]['sha256_sorted_npis']==hashlib.sha256('\n'.join(str(x) for x in sorted(s)).encode()).hexdigest(), f'list {k} does not match the commitment'
    x=len(s&new_npis); exp=K*n/N; p=float(hypergeom.sf(x-1,N,K,n)) if K>0 else None
    rep['lists'][k]={'size':n,'hits':x,'expected_by_chance':exp,'lift':(x/exp if exp>0 else None),'p_one_sided':p,'tested':k in ('top1pct','top5pct'),'confirmed_at_alpha_0.01':(p is not None and p<0.01) if k in ('top1pct','top5pct') else None}
out=os.path.join(PRO,f'check_{a.name}_{time.strftime("%Y%m%d")}.json'); json.dump(rep,open(out,'w'),indent=1); print(json.dumps(rep,indent=1))
