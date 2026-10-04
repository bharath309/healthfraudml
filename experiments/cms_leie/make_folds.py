#!/usr/bin/env python3
"""Gate 2 build: provider-disjoint folds and the pre-sampled training sets (brief §3.3), written BEFORE any model is trained.
- rows: all kept provider-years (drop_post_excl == 0), all twelve years;
- StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), y = is_fraud (primary), groups = NPI;
- training set for fold k: every positive in the training folds + a seeded random sample of negatives at RATIO per positive
  (numpy default_rng(42 + k)); the test fold is complete.
Outputs: data/folds.parquet (npi, year, fold), data/train_<k>.parquet (npi, year), out/folds_report.json."""
import os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'data'); OUT=os.path.join(HERE,'out')
YEARS=list(range(2013,2025)); RS=42
import sys
RATIO=int(sys.argv[1]) if len(sys.argv)>1 else json.load(open(os.path.join(OUT,'label_report.json')))['sampling_ratio_rule']['negatives_per_positive']; assert RATIO in (20,50)
SUF='' if RATIO==20 else f'_r{RATIO}'
parts=[]
for y in YEARS:
    t=pq.read_table(os.path.join(DATA,f'labels_{y}.parquet'),columns=['Rndrng_NPI','year','is_fraud','drop_post_excl']).to_pandas()
    t=t[t['drop_post_excl']==0]; parts.append(pd.DataFrame({'npi':t['Rndrng_NPI'].astype('int64').values,'year':t['year'].astype('int16').values,'y':t['is_fraud'].astype('int8').values}))
df=pd.concat(parts,ignore_index=True); del parts
print('rows',len(df),'positives',int(df['y'].sum()),'providers',df['npi'].nunique(),flush=True)
fp=os.path.join(DATA,'folds.parquet')
if os.path.exists(fp):   # reuse the saved assignment (identical rows/order asserted)
    f=pd.read_parquet(fp); assert len(f)==len(df) and (f['npi'].values==df['npi'].values).all() and (f['year'].values==df['year'].values).all(); df['fold']=f['fold'].values
else:
    skf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=RS)
    fold=np.full(len(df),-1,dtype='int8')
    for k,(tr,te) in enumerate(skf.split(np.zeros(len(df)),df['y'].values,df['npi'].values)): fold[te]=k+1
    assert (fold>0).all(); df['fold']=fold
# provider-disjointness
g=df.groupby('npi')['fold'].nunique(); assert (g==1).all(), 'a provider appears in more than one fold'
rep={'ratio':RATIO,'rows':int(len(df)),'positives':int(df['y'].sum()),'providers':int(df['npi'].nunique()),'folds':{}}
if not os.path.exists(fp): df[['npi','year','fold']].to_parquet(fp,index=False)
for k in range(1,6):
    tr=df[df['fold']!=k]; te=df[df['fold']==k]
    pos=tr[tr['y']==1]; neg=tr[tr['y']==0]
    rng=np.random.default_rng(RS+k); n_neg=min(len(neg),RATIO*len(pos))
    neg_s=neg.iloc[rng.choice(len(neg),size=n_neg,replace=False)]
    trs=pd.concat([pos,neg_s]).sort_values(['year','npi'])
    trs[['npi','year','y']].to_parquet(os.path.join(DATA,f'train_{k}{SUF}.parquet'),index=False)
    rep['folds'][k]={'test_rows':int(len(te)),'test_positives':int(te['y'].sum()),'test_providers':int(te['npi'].nunique()),'train_pool_rows':int(len(tr)),'train_pool_positives':int(len(pos)),
                     'train_sampled_rows':int(len(trs)),'train_sampled_negatives':int(n_neg),'train_providers':int(trs['npi'].nunique())}
    print(k,rep['folds'][k],flush=True)
json.dump(rep,open(os.path.join(OUT,f'folds_report{SUF}.json'),'w'),indent=1)
