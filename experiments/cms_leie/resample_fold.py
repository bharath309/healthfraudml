#!/usr/bin/env python3
"""Step: SMOTE-ENN (random_state=42, as run_comparison.run_experiments) on the pre-sampled training set of fold k; cached to
data/resampled_<set>_<label>_<k>.npz so every model trains on exactly the same resampled rows. Usage: resample_fold.py k [set] [label] [ratio]"""
import os, sys, time, json, numpy as np, cmsarm
from imblearn.combine import SMOTEENN
k=int(sys.argv[1]); setname=sys.argv[2] if len(sys.argv)>2 else 'primary'; label=sys.argv[3] if len(sys.argv)>3 else 'is_fraud'; ratio=int(sys.argv[4]) if len(sys.argv)>4 else 20
suffix='' if ratio==20 else f'_r{ratio}'
out=os.path.join(cmsarm.DATA,f'resampled_{setname}_{label}_{k}{suffix}.npz')
if os.path.exists(out): print('exists',out); sys.exit(0)
t0=time.time(); X,y,n,yr=cmsarm.load_train(k,setname,label,ratio)
Xr,yr_=SMOTEENN(random_state=42).fit_resample(X,y)
np.savez_compressed(out,X=Xr.astype('float32'),y=yr_.astype('int8'),n_before=len(y),pos_before=int(y.sum()))
rec={'fold':k,'set':setname,'label':label,'ratio':ratio,'rows_before':int(len(y)),'pos_before':int(y.sum()),'rows_after':int(len(yr_)),'pos_after':int(yr_.sum()),'seconds':round(time.time()-t0)}
json.dump(rec,open(os.path.join(cmsarm.OUT,f'resample_{setname}_{label}_{k}{suffix}.json'),'w'))
print(rec,flush=True)
