#!/usr/bin/env python3
"""Review-capacity table for the CMS-LEIE arm, computed only from saved outputs (no model is trained):
results/oof/<tag>_<model>_fold<k>.npy (out-of-fold probabilities, order = results/oof/keys_fold<k>.parquet),
data/test_<set>_<k>_y_<label>.npy, data/labels_<year>.parquet (excl_year) and data/by_provider_<year>.parquet (Tot_Mdcr_Pymt_Amt).
Frozen definitions (5 Oct 2026):
- population: every kept provider-year of the arm (13,527,801 rows; 11,169 fraud rows from 2,131 NPIs for the primary label);
- capacity is applied WITHIN each calendar year: rows of year y are ordered by (score descending, NPI ascending) and the top
  floor(c * n_y) rows are flagged, c in {0.5%, 1%, 2%, 5%, 10%}; a fixed budget of 1,000 rows per year is also reported;
- scores: p = model probability; p x payment = p * Tot_Mdcr_Pymt_Amt; payment = Tot_Mdcr_Pymt_Amt alone;
- per (score, capacity): flagged rows; fraud rows caught; recall_rows = caught / fraud rows; precision; lift = precision /
  prevalence; distinct later-excluded NPIs caught and their share of all matched NPIs; Medicare payments (Tot_Mdcr_Pymt_Amt)
  of the fraud rows caught and dollar-weighted recall = that sum / payments of all fraud rows; payments of all flagged rows;
- a pooled variant (all years ranked together) is reported for reconciliation only.
'Dollars' are Medicare payments to providers in their pre-exclusion years, not proven improper payments.
Usage: capacity_table.py [--set primary] [--label is_fraud] [--ratio 20] [--models gb,rf]"""
import os, json, argparse, numpy as np, pandas as pd, pyarrow.parquet as pq
ap=argparse.ArgumentParser(); ap.add_argument('--set',default='primary'); ap.add_argument('--label',default='is_fraud'); ap.add_argument('--ratio',type=int,default=20); ap.add_argument('--models',default='gb,rf'); a=ap.parse_args()
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'data'); OOF=os.path.join(HERE,'results','oof')
tag=f'{a.set}_{a.label}'+('' if a.ratio==20 else f'_r{a.ratio}'); RES=os.path.join(HERE,'results','capacity'); os.makedirs(RES,exist_ok=True)
YEARS=list(range(2013,2025)); CAPS=[0.005,0.01,0.02,0.05,0.10]; MODELS=a.models.split(',')
# 1. keys + labels + scores, in fold order
parts=[]
for k in range(1,6):
    keys=pd.read_parquet(os.path.join(OOF,f'keys_fold{k}.parquet')); keys['y']=np.load(os.path.join(DATA,f'test_{a.set}_{k}_y_{a.label}.npy'))
    for m in MODELS:
        p=np.load(os.path.join(OOF,f'{tag}_{m}_fold{k}.npy')); assert len(p)==len(keys); keys[f'p_{m}']=p.astype('float64')
    parts.append(keys)
d=pd.concat(parts,ignore_index=True); del parts; d['npi']=d['npi'].astype('int64'); d['year']=d['year'].astype('int16')
# 2. payment and exclusion year, per year
ex=[]
for y in YEARS:
    lab=pq.read_table(os.path.join(DATA,f'labels_{y}.parquet'),columns=['Rndrng_NPI','year','drop_post_excl','excl_year']).to_pandas(); lab=lab[lab['drop_post_excl']==0]
    lab['npi']=pd.to_numeric(lab['Rndrng_NPI']).astype('int64')
    bp=pq.read_table(os.path.join(DATA,f'by_provider_{y}.parquet'),columns=['Rndrng_NPI','Tot_Mdcr_Pymt_Amt']).to_pandas(); bp['npi']=pd.to_numeric(bp['Rndrng_NPI']).astype('int64'); bp=bp.drop_duplicates('npi')
    ex.append(lab[['npi','year','excl_year']].merge(bp[['npi','Tot_Mdcr_Pymt_Amt']],on='npi',how='left'))
t=pd.concat(ex,ignore_index=True); t['year']=t['year'].astype('int16'); d=d.merge(t,on=['npi','year'],how='left'); del t, ex
d['pymt']=pd.to_numeric(d['Tot_Mdcr_Pymt_Amt'],errors='coerce').fillna(0.0).astype('float64')
N=len(d); P=int(d['y'].sum()); prev=P/N; fraud_npis=int(d.loc[d['y']==1,'npi'].nunique()); fraud_dollars=float(d.loc[d['y']==1,'pymt'].sum())
d=d.sort_values(['year','npi'],kind='stable').reset_index(drop=True)   # deterministic base order before ranking
n_year=d.groupby('year')['year'].transform('size').values
def flags_within_year(score):
    """boolean masks for each capacity, ranking within year by (score desc, npi asc)."""
    order=np.lexsort((d['npi'].values,-score,d['year'].values))   # primary key year asc, then score desc, then npi asc
    rank=np.empty(N,dtype='int64');
    # rank within year = position within the year's block
    yrs_sorted=d['year'].values[order]; starts=np.r_[0,np.flatnonzero(np.diff(yrs_sorted))+1]; pos=np.arange(N)-np.repeat(starts,np.diff(np.r_[starts,N]))
    rank[order]=pos+1
    out={}
    for c in CAPS: out[str(c)]=rank<=np.floor(c*n_year)
    out['top1000_per_year']=rank<=1000
    return out
def summarise(mask):
    f=mask; nf=int(f.sum()); tp=int((f&(d['y'].values==1)).sum()); caught=d.loc[f&(d['y']==1)]
    return {'flagged_rows':nf,'fraud_rows_caught':tp,'recall_rows':tp/P,'precision':tp/nf if nf else 0.0,'lift':(tp/nf)/prev if nf else 0.0,
            'excluded_npis_caught':int(caught['npi'].nunique()),'share_of_excluded_npis':caught['npi'].nunique()/fraud_npis,
            'fraud_dollars_caught_usd':float(caught['pymt'].sum()),'dollar_weighted_recall':float(caught['pymt'].sum())/fraud_dollars,'all_flagged_dollars_usd':float(d.loc[f,'pymt'].sum())}
res={'tag':tag,'definition':'capacity within calendar year; order (score desc, NPI asc); floor(c*n_year) rows; see docstring','rows':N,'fraud_rows':P,'prevalence':prev,'fraud_npis':fraud_npis,'fraud_rows_total_payment_usd':fraud_dollars,'models':{}}
for m in MODELS:
    p=d[f'p_{m}'].values; res['models'][m]={}
    scores={'p':p,'p_x_payment':p*d['pymt'].values,'payment':d['pymt'].values} if m==MODELS[0] else {'p':p}
    for sname,s in scores.items():
        fl=flags_within_year(s); res['models'][m][sname]={c:summarise(mask) for c,mask in fl.items()}
    # pooled variant (reconciliation only): all years ranked together by (p desc, year asc, npi asc)
    order=np.lexsort((d['npi'].values,d['year'].values,-p)); rank=np.empty(N,dtype='int64'); rank[order]=np.arange(1,N+1)
    res['models'][m]['pooled_p']={**{str(c):summarise(rank<=int(np.floor(c*N))) for c in CAPS},'top1000':summarise(rank<=1000),'top5000':summarise(rank<=5000)}
json.dump(res,open(os.path.join(RES,f'{tag}.json'),'w'),indent=1)
L=[f'# Review-capacity table ({tag}); capacity applied within each calendar year; from saved OOF outputs only',
   f'Rows {N:,}; fraud rows {P:,} ({100*prev:.4f}%) from {fraud_npis:,} NPIs; Medicare payments of fraud rows ${fraud_dollars/1e6:,.1f} M.','',
   '| model | score | capacity | flagged | fraud rows caught | recall (rows) | precision | lift | excluded NPIs (share) | fraud $ (M) | $-weighted recall | all flagged $ (M) |','|---|---|---|---|---|---|---|---|---|---|---|---|']
for m in MODELS:
    for sname,v in res['models'][m].items():
        if sname.startswith('pooled'): continue
        for c,r in v.items():
            cap='1,000/yr' if c=='top1000_per_year' else f'top {float(c)*100:g}%'
            L.append(f"| {m} | {sname} | {cap} | {r['flagged_rows']:,} | {r['fraud_rows_caught']:,} | {r['recall_rows']:.3f} | {r['precision']:.4f} | {r['lift']:.1f}× | {r['excluded_npis_caught']:,} ({r['share_of_excluded_npis']:.3f}) | {r['fraud_dollars_caught_usd']/1e6:,.1f} | {r['dollar_weighted_recall']:.3f} | {r['all_flagged_dollars_usd']/1e6:,.1f} |")
L+=['','Pooled ranking (reconciliation only), score p:','| model | capacity | flagged | fraud rows caught | recall (rows) | precision | lift |','|---|---|---|---|---|---|---|']
for m in MODELS:
    for c,r in res['models'][m]['pooled_p'].items(): L.append(f"| {m} | {c if c.startswith('top') else 'top '+str(float(c)*100)+'%'} | {r['flagged_rows']:,} | {r['fraud_rows_caught']:,} | {r['recall_rows']:.3f} | {r['precision']:.4f} | {r['lift']:.1f}× |")
open(os.path.join(RES,f'{tag}.md'),'w').write('\n'.join(L)+'\n'); print('\n'.join(L))
