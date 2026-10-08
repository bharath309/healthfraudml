#!/usr/bin/env python3
"""Paper T — POST-HOC, DESCRIPTIVE (not pre-specified in Protocol T): timing of the test-year positives' exclusions relative to the
as-of date, and the split of the 1 % flags into providers already excluded at 1 Jan t and providers excluded later.
Reads data/asof_known.parquet (labels; asof_labels.py) and results/asof/scores_<t>.parquet (asof_train.py) and the hindsight OOF
files; writes results/asof/posthoc_timing.json (counts and shares only; no NPI). Nothing here changes any pre-specified rule.
Usage: paperT_posthoc_timing.py ARM_DIR  (ARM_DIR = the cms_leie folder with data/ and results/)"""
import os, sys, json, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
ARM=sys.argv[1]; DATA=os.path.join(ARM,'data'); RES=os.path.join(ARM,'results','asof'); T=[2018,2019,2020,2021,2022]; K_BUDGET=0.01
K=pq.read_table(os.path.join(DATA,'asof_known.parquet'),columns=['npi','year','is_fraud','excl_year','excl_end_date']).to_pandas()
def hindsight_scores(t):
    """Companion-study out-of-fold GB probabilities for year-t rows (results/oof_fold<k>.parquet: npi, year, p_gb, p_rf, is_fraud),
    read exactly as asof_metrics.py reads them."""
    import glob
    O=pd.concat([pd.read_parquet(f,columns=['npi','year','p_gb']) for f in sorted(glob.glob(os.path.join(ARM,'results','oof_fold*.parquet')))],ignore_index=True)
    O=O[O['year']==t]; return O[['npi','p_gb']].rename(columns={'p_gb':'p_h'})
def topk_mask(p,npi,k):
    order=np.lexsort((npi,-p)); n=int(np.floor(k*len(p))); m=np.zeros(len(p),bool); m[order[:n]]=True; return m
out={'note':'post hoc, descriptive; not pre-specified; counts only','budget':K_BUDGET,'years':{}}
for t in T:
    Kt=K[K['year']==t]; pos=Kt[Kt['is_fraud']==1]; jan=pd.Timestamp(year=t,month=1,day=1)
    already=(pos['excl_end_date']<jan); within3=(pos['excl_year']<=t+3)&~already; later=~already&~within3
    ey=pos.loc[~already,'excl_year'].astype(int)
    S=pd.read_parquet(os.path.join(RES,f'scores_{t}.parquet'),columns=['npi','is_fraud','p_gb_A'])
    S=S.merge(Kt[['npi','excl_end_date']],on='npi',how='left'); y=S['is_fraud'].values.astype(bool); known=(S['excl_end_date']<jan).values&y
    m=topk_mask(S['p_gb_A'].values.astype('float64'),S['npi'].values,K_BUDGET)
    r={'test_rows':int(len(Kt)),'test_positives':int(len(pos)),'already_excluded_at_1jan_t':int(already.sum()),'excluded_within_t_to_t_plus_3':int(within3.sum()),
       'excluded_after_t_plus_3':int(later.sum()),'share_already_excluded':float(already.mean()),
       'excl_year_counts_not_yet_excluded':{str(k):int(v) for k,v in sorted(ey.value_counts().items())},
       'median_years_to_exclusion_not_yet_excluded':float(np.median(ey.values-t)) if len(ey) else None,
       'asof_gb_A_top1pct':{'flags':int(m.sum()),'tp':int((m&y).sum()),'tp_already_excluded':int((m&known).sum()),'tp_not_yet_excluded':int((m&y&~known).sum()),
                            'recall_all':float((m&y).sum()/y.sum()),'recall_already_excluded':float((m&known).sum()/max(1,known.sum())),
                            'recall_not_yet_excluded':float((m&y&~known).sum()/max(1,(y&~known).sum()))}}
    try:
        H=hindsight_scores(t); S2=S.merge(H,on='npi',how='left'); ph=S2['p_h'].fillna(-1).values.astype('float64'); mh=topk_mask(ph,S2['npi'].values,K_BUDGET)
        r['hindsight_gb_top1pct']={'tp':int((mh&y).sum()),'tp_already_excluded':int((mh&known).sum()),'tp_not_yet_excluded':int((mh&y&~known).sum()),
                                  'recall_all':float((mh&y).sum()/y.sum()),'recall_already_excluded':float((mh&known).sum()/max(1,known.sum())),
                                  'recall_not_yet_excluded':float((mh&y&~known).sum()/max(1,(y&~known).sum())),'rows_without_oof_score':int((S2['p_h'].isna()).sum())}
        a=r['asof_gb_A_top1pct']['recall_not_yet_excluded']; h=r['hindsight_gb_top1pct']['recall_not_yet_excluded']; r['ratio_not_yet_excluded']=(h/a) if a>0 else None
    except Exception as e:
        r['hindsight_gb_top1pct']={'error':str(e)[:200]}
    out['years'][str(t)]=r; print(t,json.dumps({k:v for k,v in r.items() if k!='excl_year_counts_not_yet_excluded'}),flush=True)
json.dump(out,open(os.path.join(RES,'posthoc_timing.json'),'w'),indent=1); print('written',os.path.join(RES,'posthoc_timing.json'))
