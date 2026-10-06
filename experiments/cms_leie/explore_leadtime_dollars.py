#!/usr/bin/env python3
"""EXPLORATORY (post-hoc, no frozen protocol, no new model): what the existing out-of-fold scores of the CMS-LEIE arm
say about (a) lead time to exclusion, (b) payment-at-risk capture at review capacity, (c) score vs years-to-exclusion,
(d) the as-of label-completeness problem. Inputs are only saved outputs of the frozen arm (results/oof/*.npy, the fold keys,
labels_<year>.parquet, by_provider_<year>.parquet). Nothing here is a confirmatory result; it is hypothesis-generating
evidence for the candidate-methods memo of 5 Oct 2026. Written 5 Oct 2026."""
import os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'data'); OOF=os.path.join(HERE,'results','oof'); OUT=os.path.join(HERE,'explore'); os.makedirs(OUT,exist_ok=True)
YEARS=list(range(2013,2025)); CAPS=[0.005,0.01,0.02,0.05,0.10]
# 1. out-of-fold scores with their keys (order = keys_fold<k>.parquet, written by train_model.py)
parts=[]
for k in range(1,6):
    keys=pd.read_parquet(os.path.join(OOF,f'keys_fold{k}.parquet'))
    for m in ['gb','rf']:
        p=np.load(os.path.join(OOF,f'primary_is_fraud_{m}_fold{k}.npy')); assert len(p)==len(keys); keys[f'p_{m}']=p
    keys['fold']=np.int8(k); parts.append(keys)
oof=pd.concat(parts,ignore_index=True); del parts
oof['npi']=oof['npi'].astype('int64'); oof['year']=oof['year'].astype('int16')
print('oof rows',len(oof),flush=True)
# 2. per-year attach labels, exclusion year, provider type and Medicare payment
rows=[]
for y in YEARS:
    lab=pq.read_table(os.path.join(DATA,f'labels_{y}.parquet'),columns=['Rndrng_NPI','year','is_fraud','excl_year','drop_post_excl','Rndrng_Prvdr_Type']).to_pandas()
    lab=lab[lab['drop_post_excl']==0]
    bp=pq.read_table(os.path.join(DATA,f'by_provider_{y}.parquet'),columns=['Rndrng_NPI','Tot_Mdcr_Pymt_Amt','Tot_Benes']).to_pandas()
    lab['npi']=pd.to_numeric(lab['Rndrng_NPI'],errors='coerce').astype('int64'); bp['npi']=pd.to_numeric(bp['Rndrng_NPI'],errors='coerce').astype('int64')
    bp=bp.drop_duplicates('npi')
    o=oof[oof['year']==y].merge(lab[['npi','is_fraud','excl_year','Rndrng_Prvdr_Type']],on='npi',how='left').merge(bp[['npi','Tot_Mdcr_Pymt_Amt','Tot_Benes']],on='npi',how='left')
    assert o['is_fraud'].notna().all(), f'unmatched labels in {y}'
    rows.append(o); print(y,len(o),int(o['is_fraud'].sum()),flush=True)
df=pd.concat(rows,ignore_index=True); del rows, oof
df['is_fraud']=df['is_fraud'].astype('int8'); df['pymt']=pd.to_numeric(df['Tot_Mdcr_Pymt_Amt'],errors='coerce').fillna(0.0)
assert len(df)==13527801 and int(df['is_fraud'].sum())==11169
res={'note':'EXPLORATORY post-hoc analysis of saved OOF scores (cloud run, GB and RF); not a frozen-protocol result','rows':int(len(df)),'fraud_rows':int(df['is_fraud'].sum()),'fraud_npis':int(df.loc[df['is_fraud']==1,'npi'].nunique())}
# 3. within-year percentile rank (1 = highest score) and top-k flags per capacity, per model
for m in ['gb','rf']:
    df[f'rank_{m}']=df.groupby('year')[f'p_{m}'].rank(ascending=False,method='first')
df['n_year']=df.groupby('year')['year'].transform('size')
for m in ['gb','rf']:
    df[f'pct_{m}']=df[f'rank_{m}']/df['n_year']
fr=df[df['is_fraud']==1].copy(); fr['yte']=(fr['excl_year']-fr['year']).astype(int)   # years to exclusion (>=1 by construction)
res['years_to_exclusion_distribution']={int(k):int(v) for k,v in fr['yte'].value_counts().sort_index().items()}
# 4. recall, precision, dollar capture at capacity, per model (capacity applied per year)
tot_fraud_dollars=float(fr['pymt'].sum()); res['fraud_rows_total_medicare_payment_usd']=tot_fraud_dollars
cap={}
for m in ['gb','rf']:
    cap[m]={}
    for c in CAPS:
        flag=df[f'pct_{m}']<=c; nf=int(flag.sum()); tp=int((flag&(df['is_fraud']==1)).sum())
        d_fraud=float(df.loc[flag&(df['is_fraud']==1),'pymt'].sum()); d_all=float(df.loc[flag,'pymt'].sum())
        npis_hit=int(df.loc[flag&(df['is_fraud']==1),'npi'].nunique())
        cap[m][str(c)]={'flagged_rows':nf,'fraud_rows_caught':tp,'recall_rows':tp/11169,'precision':tp/nf,'lift':(tp/nf)/(11169/len(df)),
                        'excluded_npis_caught':npis_hit,'recall_npis':npis_hit/res['fraud_npis'],'fraud_dollars_caught_usd':d_fraud,'dollar_weighted_recall':d_fraud/tot_fraud_dollars,
                        'all_flagged_dollars_usd':d_all,'fraud_dollars_per_flagged_row_usd':d_fraud/nf}
    # fixed count: 1,000 per year
    flag=df[f'rank_{m}']<=1000; nf=int(flag.sum()); tp=int((flag&(df['is_fraud']==1)).sum()); d_fraud=float(df.loc[flag&(df['is_fraud']==1),'pymt'].sum())
    cap[m]['top1000_per_year']={'flagged_rows':nf,'fraud_rows_caught':tp,'precision':tp/nf,'lift':(tp/nf)/(11169/len(df)),'excluded_npis_caught':int(df.loc[flag&(df['is_fraud']==1),'npi'].nunique()),'fraud_dollars_caught_usd':d_fraud,'dollar_weighted_recall':d_fraud/tot_fraud_dollars}
res['capacity']=cap
# 4b. alternative triage rules at the same capacity (GB): expected payments at risk = p x Medicare payment; payment alone (the 'audit the biggest billers' rule)
alt={}
for name,score in [('p_gb',df['p_gb']),('p_gb_x_payment',df['p_gb']*df['pymt']),('payment_only',df['pymt'])]:
    r=score.groupby(df['year']).rank(ascending=False,method='first')/df['n_year']; alt[name]={}
    for c in CAPS:
        flag=r<=c; nf=int(flag.sum()); tp=int((flag&(df['is_fraud']==1)).sum()); d_fraud=float(df.loc[flag&(df['is_fraud']==1),'pymt'].sum())
        alt[name][str(c)]={'flagged_rows':nf,'fraud_rows_caught':tp,'recall_rows':tp/11169,'excluded_npis_caught':int(df.loc[flag&(df['is_fraud']==1),'npi'].nunique()),'fraud_dollars_caught_usd':d_fraud,'dollar_weighted_recall':d_fraud/tot_fraud_dollars}
res['triage_rules_gb']=alt
# 4c. payments to later-excluded providers after the first year they entered the GB top 1% of their year
ff=fr[fr['pct_gb']<=0.01].groupby('npi')['year'].min().rename('first_flag'); frf=fr.merge(ff,on='npi',how='inner')
res['payments_after_first_flag_gb_top1pct']={'flagged_excluded_npis':int(len(ff)),'usd_strictly_after_flag_year':float(frf.loc[frf['year']>frf['first_flag'],'pymt'].sum()),'usd_from_flag_year_on':float(frf.loc[frf['year']>=frf['first_flag'],'pymt'].sum()),
    'npis_with_later_years':int(frf.loc[frf['year']>frf['first_flag'],'npi'].nunique()),'median_later_billing_years':float(frf.loc[frf['year']>frf['first_flag']].groupby('npi').size().median())}
# 4d. within-provider trend of the GB within-year percentile over the fraud-labelled years (escalation check), excluded NPIs with >=3 rows vs a seeded sample of non-fraud NPIs
def slopes(sub):
    g=sub.groupby('npi'); return g.apply(lambda x: np.polyfit(x['year'].values.astype(float),x['pct_gb'].values,1)[0] if x['year'].nunique()>1 else np.nan,include_groups=False).dropna()
cnt=fr.groupby('npi')['year'].transform('size'); s1=slopes(fr[cnt>=3])
nf=df[df['is_fraud']==0]; c0=nf.groupby('npi')['year'].size(); el=c0[c0>=3].index.values; rng=np.random.default_rng(0); samp=rng.choice(el,size=20000,replace=False); s0=slopes(nf[nf['npi'].isin(samp)])
res['within_provider_score_trend']={'excluded_npis_ge3_rows':int(len(s1)),'share_rank_improving_toward_exclusion':float((s1<0).mean()),'median_slope_per_year':float(s1.median()),
    'nonfraud_sample_npis':int(len(s0)),'nonfraud_share_rank_improving':float((s0<0).mean()),'nonfraud_median_slope':float(s0.median())}
# 4e. equal-share-per-specialty allocation (top k% within each year x provider type) vs global ranking
df['pct_gb_type']=df.groupby(['year','Rndrng_Prvdr_Type'])['p_gb'].rank(ascending=False,method='first')/df.groupby(['year','Rndrng_Prvdr_Type'])['p_gb'].transform('size')
res['equal_share_per_type_allocation_gb']={str(c):{'flagged_rows':int((df['pct_gb_type']<=c).sum()),'fraud_rows_caught':int(((df['pct_gb_type']<=c)&(df['is_fraud']==1)).sum()),'global_fraud_rows_caught':int(((df['pct_gb']<=c)&(df['is_fraud']==1)).sum())} for c in [0.01,0.05]}
# 5. lead time: per excluded NPI, first year flagged at capacity c (GB) -> lead = excl_year - first flagged year
lead={}
for m in ['gb','rf']:
    lead[m]={}
    for c in [0.01,0.05,0.10]:
        f=fr[fr[f'pct_{m}']<=c].groupby('npi').agg(first_flag=('year','min'),excl=('excl_year','first'))
        f['lead']=(f['excl']-f['first_flag']).astype(int)
        n=res['fraud_npis']; lead[m][str(c)]={'npis_flagged_ever':int(len(f)),'share_of_excluded_npis':len(f)/n,'lead_years_median':float(f['lead'].median()) if len(f) else None,
            'lead_ge2_share_of_all_excluded':float((f['lead']>=2).sum()/n),'lead_ge3_share_of_all_excluded':float((f['lead']>=3).sum()/n),'lead_distribution':{int(k):int(v) for k,v in f['lead'].value_counts().sort_index().items()}}
res['lead_time']=lead
# 6. score vs years-to-exclusion (GB): mean within-year percentile and share in top 1% by yte
s=fr.groupby('yte').agg(n=('npi','size'),mean_pct_gb=('pct_gb','mean'),top1_gb=('pct_gb',lambda v:float((v<=0.01).mean())),top5_gb=('pct_gb',lambda v:float((v<=0.05).mean())),mean_pct_rf=('pct_rf','mean'))
res['score_vs_years_to_exclusion']={int(k):{kk:float(vv) for kk,vv in r.items()} for k,r in s.iterrows()}
# 7. empirical exclusion lag: years of billing observed before the exclusion year, per excluded NPI
span=fr.groupby('npi').agg(first=('year','min'),last=('year','max'),excl=('excl_year','first'))
res['billing_years_before_exclusion']={'median_first_to_excl':float((span['excl']-span['first']).median()),'median_last_to_excl':float((span['excl']-span['last']).median()),
    'dist_last_to_excl':{int(k):int(v) for k,v in (span['excl']-span['last']).astype(int).value_counts().sort_index().items()}}
# 8. as-of label completeness: positives among rows with year < t that are KNOWN at t (excl_year <= t) vs eventual
asof={}
for t in range(2015,2026):
    sub=fr[fr['year']<t]; asof[t]={'eventual_positive_rows':int(len(sub)),'known_at_t':int((sub['excl_year']<=t).sum()),'share_known':float((sub['excl_year']<=t).mean()) if len(sub) else None}
res['as_of_label_completeness']=asof
# 9. where the flags fall (aggregate provider types only; no provider is identified)
top=df[df['pct_gb']<=0.01]
pt_flag=top['Rndrng_Prvdr_Type'].value_counts(normalize=True).head(8); pt_fraud=fr['Rndrng_Prvdr_Type'].value_counts(normalize=True).head(8); pt_all=df['Rndrng_Prvdr_Type'].value_counts(normalize=True).head(8)
res['provider_type_shares']={'top1pct_flags':{k:float(v) for k,v in pt_flag.items()},'fraud_rows':{k:float(v) for k,v in pt_fraud.items()},'all_rows':{k:float(v) for k,v in pt_all.items()}}
# per-type precision at top 1% (types with >= 2,000 flagged rows)
g=top.groupby('Rndrng_Prvdr_Type')['is_fraud'].agg(['size','sum']); g=g[g['size']>=2000]; g['precision']=g['sum']/g['size']
res['top1pct_precision_by_type']={k:{'flagged':int(r['size']),'fraud':int(r['sum']),'precision':float(r['precision'])} for k,r in g.sort_values('precision',ascending=False).iterrows()}
json.dump(res,open(os.path.join(OUT,'leadtime_dollars_Oct5.json'),'w'),indent=1,default=float)
# markdown
L=['# EXPLORATORY: lead time, dollar capture, score vs years-to-exclusion (cloud OOF, GB and RF) — 5 Oct 2026','Post-hoc analysis of saved out-of-fold scores; not a frozen-protocol result. Capacity is applied within each calendar year.','',
   f"Fraud rows {res['fraud_rows']} from {res['fraud_npis']} excluded NPIs; their Medicare payments total ${tot_fraud_dollars/1e6:,.1f} M.",'',
   '| model | capacity | flagged rows | fraud rows caught | recall (rows) | precision | lift | excluded NPIs caught (share) | fraud $ caught (M) | $-weighted recall | all flagged $ (M) |','|---|---|---|---|---|---|---|---|---|---|---|']
for m in ['gb','rf']:
    for c,v in cap[m].items():
        if c=='top1000_per_year': L.append(f"| {m.upper()} | 1,000/yr | {v['flagged_rows']:,} | {v['fraud_rows_caught']:,} | {v['fraud_rows_caught']/11169:.3f} | {v['precision']:.4f} | {v['lift']:.1f}× | {v['excluded_npis_caught']:,} ({v['excluded_npis_caught']/res['fraud_npis']:.3f}) | {v['fraud_dollars_caught_usd']/1e6:,.1f} | {v['dollar_weighted_recall']:.3f} | — |")
        else: L.append(f"| {m.upper()} | top {float(c)*100:g}% | {v['flagged_rows']:,} | {v['fraud_rows_caught']:,} | {v['recall_rows']:.3f} | {v['precision']:.4f} | {v['lift']:.1f}× | {v['excluded_npis_caught']:,} ({v['recall_npis']:.3f}) | {v['fraud_dollars_caught_usd']/1e6:,.1f} | {v['dollar_weighted_recall']:.3f} | {v['all_flagged_dollars_usd']/1e6:,.1f} |")
L+=['','## Triage rules at the same capacity (GB): probability vs expected payments at risk (p × payment) vs payment alone','| rule | capacity | fraud rows caught | recall (rows) | excluded NPIs caught | fraud $ caught (M) | $-weighted recall |','|---|---|---|---|---|---|---|']
for name,v in alt.items():
    for c,r in v.items(): L.append(f"| {name} | top {float(c)*100:g}% | {r['fraud_rows_caught']:,} | {r['recall_rows']:.3f} | {r['excluded_npis_caught']:,} | {r['fraud_dollars_caught_usd']/1e6:,.1f} | {r['dollar_weighted_recall']:.3f} |")
pa=res['payments_after_first_flag_gb_top1pct']; L+=['',f"Payments to later-excluded providers after the first year they entered the GB top 1%: {pa['flagged_excluded_npis']} NPIs; ${pa['usd_strictly_after_flag_year']/1e6:,.1f} M in fraud-labelled years strictly after the flag year (${pa['usd_from_flag_year_on']/1e6:,.1f} M from the flag year on); {pa['npis_with_later_years']} of them billed in later years, median {pa['median_later_billing_years']} further years."]
wt=res['within_provider_score_trend']; L+=[f"Within-provider trend of the GB percentile (escalation check): excluded NPIs with ≥3 rows {wt['excluded_npis_ge3_rows']}, share whose rank improves toward exclusion {wt['share_rank_improving_toward_exclusion']:.3f} (median slope {wt['median_slope_per_year']:.4f}/yr) vs non-fraud sample {wt['nonfraud_share_rank_improving']:.3f} ({wt['nonfraud_median_slope']:.4f}/yr) — no escalation signal in the current score."]
eq=res['equal_share_per_type_allocation_gb']; L+=[f"Equal-share-per-specialty allocation (top k% within each year × provider type) vs global ranking, fraud rows caught: 1% {eq['0.01']['fraud_rows_caught']:,} vs {eq['0.01']['global_fraud_rows_caught']:,}; 5% {eq['0.05']['fraud_rows_caught']:,} vs {eq['0.05']['global_fraud_rows_caught']:,}."]
L+=['','## Lead time to exclusion (first year the provider entered the top-k of its year)','| model | capacity | excluded NPIs flagged ever | share | median lead (yr) | flagged ≥2 yr before exclusion (share of all excluded) | ≥3 yr |','|---|---|---|---|---|---|---|']
for m in ['gb','rf']:
    for c,v in lead[m].items(): L.append(f"| {m.upper()} | top {float(c)*100:g}% | {v['npis_flagged_ever']:,} | {v['share_of_excluded_npis']:.3f} | {v['lead_years_median']} | {v['lead_ge2_share_of_all_excluded']:.3f} | {v['lead_ge3_share_of_all_excluded']:.3f} |")
L+=['','## Score vs years to exclusion (fraud rows; GB)','| years to exclusion | rows | mean within-year percentile (GB) | share in top 1% (GB) | share in top 5% (GB) | mean percentile (RF) |','|---|---|---|---|---|---|']
for k,r in res['score_vs_years_to_exclusion'].items(): L.append(f"| {k} | {int(r['n']):,} | {r['mean_pct_gb']:.3f} | {r['top1_gb']:.3f} | {r['top5_gb']:.3f} | {r['mean_pct_rf']:.3f} |")
b=res['billing_years_before_exclusion']; L+=['',f"Billing years observed before exclusion: median first-observed-to-exclusion {b['median_first_to_excl']} yr; median last-observed-to-exclusion {b['median_last_to_excl']} yr; distribution of last-to-exclusion: {b['dist_last_to_excl']}.",'',
   '## As-of label completeness (positives among rows with year < t that are already excluded at t)','| t | eventual positive rows (year < t) | known at t | share known |','|---|---|---|---|']
for t,v in asof.items(): L.append(f"| {t} | {v['eventual_positive_rows']:,} | {v['known_at_t']:,} | {v['share_known']:.3f} |" if v['share_known'] is not None else f"| {t} | 0 | 0 | — |")
L+=['','## Where the top-1% flags fall (aggregate provider types; precision at top 1%, types with ≥2,000 flagged rows)','| provider type | flagged | fraud | precision |','|---|---|---|---|']
for k,v in res['top1pct_precision_by_type'].items(): L.append(f"| {k} | {v['flagged']:,} | {v['fraud']:,} | {v['precision']:.4f} |")
open(os.path.join(OUT,'leadtime_dollars_Oct5.md'),'w').write('\n'.join(L)+'\n'); print('\n'.join(L))
