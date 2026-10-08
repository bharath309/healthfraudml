#!/usr/bin/env python3
"""Protocol T step 2: build the training and test matrices for as-of year t and feature variant A or B.
Usage: asof_build.py T VARIANT        (VARIANT in {A, B})
A: arm features; RobustScaler statistics (median/IQR, IQR 0 -> 1) computed on ALL kept rows with year <= t-1, applied to the
   sampled training rows and to year t.   B: peer-normalised: the 48 numeric columns are (x - median_{year,type})/IQR_{year,type}
   within the row's own calendar year and provider type (>= 30 rows, else the year's all-type statistics); indicators and
   categoricals unchanged; no labels used.
Training sample: all known_t positives + 20 negatives per positive, default_rng(42 + t), from rows with year <= t-1; then
SMOTEENN(random_state=42). Outputs (data/): asof_train_<t>_<V>.parquet (keys), asof_resampled_<t>_<V>.npz, asof_test_<t>_<V>.npz,
and out/asof_scaler_<t>_<V>.json. Resumable: existing outputs are not rebuilt."""
import os, sys, json, time, numpy as np, pandas as pd
from imblearn.combine import SMOTEENN
from asof_common import DATA, OUT, FEATS, NUMERIC, PTYPE_COL, PAY_COL, known_table, year_rows, write_json, sha256, SMOKE
t=int(sys.argv[1]); V=sys.argv[2]; assert V in ('A','B'); T0=time.time()
tag=f'{t}_{V}'; res_npz=os.path.join(DATA,f'asof_resampled_{tag}.npz'); test_npz=os.path.join(DATA,f'asof_test_{tag}.npz'); keys_pq=os.path.join(DATA,f'asof_train_{tag}.parquet')
if os.path.exists(res_npz) and os.path.exists(test_npz): print('exists',res_npz,test_npz); sys.exit(0)
K=known_table(); kn=f'known_{t}'
tr=K[K['year']<=t-1]; posk=tr[tr[kn]==1]; negk=tr[tr[kn]==0]
rng=np.random.default_rng(42+t); nneg=min(len(negk),20*len(posk)); negs=negk.iloc[rng.choice(len(negk),size=nneg,replace=False)]
keys=pd.concat([posk[['npi','year']].assign(y=1),negs[['npi','year']].assign(y=0)],ignore_index=True)
keys.to_parquet(keys_pq,index=False); print(f'train keys: {len(posk)} known positives + {nneg} negatives from {len(tr)} rows <= {t-1}',flush=True)
# ---- scaling statistics
numeric_idx=[FEATS.index(c) for c in NUMERIC]
def robust_stats(df,cols):
    q=df[cols].quantile([0.25,0.5,0.75]); med=q.loc[0.5]; iqr=(q.loc[0.75]-q.loc[0.25]).replace(0,1.0); return med.astype('float64'),iqr.astype('float64')
years_tr=sorted(tr['year'].unique().tolist())
if V=='A':
    # column-wise quantiles over all kept rows with year <= t-1 (one feature column at a time across the years; numerically the
    # same pandas quantile([0.25,0.5,0.75]) of the concatenated rows as the first implementation, without holding all years in memory)
    import pyarrow.parquet as _pq
    med={}; iqr={}
    for c in FEATS:
        col=pd.concat([_pq.read_table(os.path.join(DATA,f'feat_{y}.parquet'),columns=[c]).to_pandas()[c] for y in years_tr],ignore_index=True)
        q=col.quantile([0.25,0.5,0.75]); m=float(q.loc[0.5]); i=float(q.loc[0.75]-q.loc[0.25]); iqr[c]=1.0 if i==0 else i; med[c]=m; del col
    med=pd.Series(med)[FEATS].astype('float64'); iqr=pd.Series(iqr)[FEATS].astype('float64')
    stats={c:{'median':float(med[c]),'iqr':float(iqr[c])} for c in FEATS}
    def scale(df): return ((df[FEATS].astype('float64')-med.values)/iqr.values).to_numpy(dtype='float32')
else:
    stats={}
    def scale(df):
        """df has FEATS + 'year'; peer-normalise NUMERIC within (year, provider type); others unchanged (float32)."""
        X=df[FEATS].astype('float64').copy()
        for y,g in df.groupby('year'):
            full=year_rows(y,FEATS+[PTYPE_COL] if PTYPE_COL not in FEATS else FEATS)        # statistics from ALL rows of year y
            medA,iqrA=robust_stats(full,NUMERIC)
            grp=full.groupby(PTYPE_COL); cnt=grp.size()
            q=grp[NUMERIC].quantile([0.25,0.5,0.75]).unstack(level=-1)   # index: type; columns: (col, q)
            m_t=q.xs(0.5,axis=1,level=1); i_t=(q.xs(0.75,axis=1,level=1)-q.xs(0.25,axis=1,level=1)).replace(0,1.0)
            small=cnt[cnt<30].index; m_t.loc[small,:]=medA.values; i_t.loc[small,:]=iqrA.values
            types=g[PTYPE_COL].astype(int).values
            m=m_t.reindex(types).fillna(medA).to_numpy(); i=i_t.reindex(types).fillna(iqrA).to_numpy()
            X.loc[g.index,NUMERIC]=(g[NUMERIC].astype('float64').to_numpy()-m)/i
            stats[str(y)]={'types':int(len(cnt)),'types_small':int(len(small))}
        return X.to_numpy(dtype='float32')
write_json({'t':t,'variant':V,'years_train':years_tr,'stats':stats,'protocol':'robust (x-median)/IQR, IQR 0->1','rows_for_stats':'all kept rows with year<=t-1' if V=='A' else 'per (year, provider type)'},os.path.join(OUT,f'asof_scaler_{tag}.json'))
# ---- training matrix
Xtr=[];ytr=[]
for y in years_tr:
    kk=keys[keys['year']==y]
    if len(kk)==0: continue
    f=year_rows(y,FEATS).set_index('npi').loc[kk['npi'].values]; f['year']=y; f=f.reset_index()
    Xtr.append(scale(f)); ytr.append(kk['y'].to_numpy(dtype='int8'))
X=np.vstack(Xtr); y=np.concatenate(ytr); print('train matrix',X.shape,'positives',int(y.sum()),f'{time.time()-T0:.0f}s',flush=True)
sm=SMOTEENN(random_state=42) if not SMOKE else SMOTEENN(random_state=42,smote=__import__('imblearn.over_sampling',fromlist=['SMOTE']).SMOTE(random_state=42,k_neighbors=3))
Xr,yr=sm.fit_resample(X.astype('float64'),y); np.savez_compressed(res_npz,X=Xr.astype('float32'),y=yr.astype('int8'))
print('resampled',Xr.shape,'positives',int(yr.sum()),f'{time.time()-T0:.0f}s',flush=True)
# ---- test matrix: every kept row of year t
te=K[K['year']==t][['npi','is_fraud',f'is_fraud_3yr_{t}']]
f=year_rows(t,FEATS+([PAY_COL] if PAY_COL not in FEATS else [])).set_index('npi').loc[te['npi'].values]; f['year']=t; f=f.reset_index()
Xte=scale(f); np.savez_compressed(test_npz,X=Xte,y=te['is_fraud'].to_numpy(dtype='int8'),y3=te[f'is_fraud_3yr_{t}'].to_numpy(dtype='int8'),npi=te['npi'].to_numpy(dtype='int64'),
                  pay=f[PAY_COL].to_numpy(dtype='float64'),ptype=f[PTYPE_COL].to_numpy(dtype='int32'))
print('test matrix',Xte.shape,'positives',int(te['is_fraud'].sum()),'| sha256',sha256(res_npz)[:12],sha256(test_npz)[:12],f'{time.time()-T0:.0f}s')
# ---- previous-year matrix (year t-1, same scaling) for the yield-aware allocation of Protocol T §6
prev_npz=os.path.join(DATA,f'asof_prev_{tag}.npz'); pv=K[K['year']==t-1][['npi']]
f=year_rows(t-1,FEATS+([PAY_COL] if PAY_COL not in FEATS else [])).set_index('npi').loc[pv['npi'].values]; f['year']=t-1; f=f.reset_index()
Xpv=scale(f); np.savez_compressed(prev_npz,X=Xpv,npi=pv['npi'].to_numpy(dtype='int64'),ptype=f[PTYPE_COL].to_numpy(dtype='int32'))
print('prev-year matrix',Xpv.shape,f'{time.time()-T0:.0f}s')
