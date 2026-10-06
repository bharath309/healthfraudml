#!/usr/bin/env python3
"""Label completeness over time for the CMS-LEIE arm, from the label files only (identical on every machine):
for each model-build year t, the share of the eventual fraud-labelled provider-years with year < t whose exclusion year
(as in labels_<year>.parquet, LEIE of 10 Sep 2026) is <= t, i.e. the share a model built at the start of year t could have
known. Also the distribution of years-to-exclusion and of observed billing years before exclusion. Writes out/label_completeness.json."""
import os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'data'); OUT=os.path.join(HERE,'out')
parts=[]
for y in range(2013,2025):
    t=pq.read_table(os.path.join(DATA,f'labels_{y}.parquet'),columns=['Rndrng_NPI','year','is_fraud','drop_post_excl','excl_year']).to_pandas()
    t=t[(t['drop_post_excl']==0)&(t['is_fraud']==1)]; parts.append(pd.DataFrame({'npi':t['Rndrng_NPI'].astype('int64').values,'year':t['year'].astype(int).values,'excl_year':t['excl_year'].astype(int).values}))
fr=pd.concat(parts,ignore_index=True); assert len(fr)==11169 and fr['npi'].nunique()==2131
res={'fraud_rows':int(len(fr)),'fraud_npis':int(fr['npi'].nunique()),'as_of':{},'years_to_exclusion':{int(k):int(v) for k,v in (fr['excl_year']-fr['year']).value_counts().sort_index().items()}}
for t in range(2015,2026):
    sub=fr[fr['year']<t]; res['as_of'][t]={'eventual_positive_rows_before_t':int(len(sub)),'known_at_t':int((sub['excl_year']<=t).sum()),'share_known':float((sub['excl_year']<=t).mean())}
span=fr.groupby('npi').agg(first=('year','min'),last=('year','max'),excl=('excl_year','first'))
res['billing_years_before_exclusion']={'median_first_observed_to_exclusion':float((span['excl']-span['first']).median()),'median_last_observed_to_exclusion':float((span['excl']-span['last']).median())}
json.dump(res,open(os.path.join(OUT,'label_completeness.json'),'w'),indent=1); print(json.dumps(res['as_of'],indent=0)[:600]); print(res['billing_years_before_exclusion'])
