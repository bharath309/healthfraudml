#!/usr/bin/env python3
"""Adds the three entries the sensitivities need but build_features.py does not write (Cowork added them by hand on
3 Oct, 17:30 and 18:30 UTC): scaler statistics for `cc_missing` (2017-2024 kept rows) and `year` (all kept rows), and
the `primary_plus_year` feature set (primary + year). Same RobustScaler convention as build_features.py. Idempotent."""
import os, json, numpy as np, pyarrow.parquet as pq
import cmsarm
OUT, DATA = cmsarm.OUT, cmsarm.DATA
sp, fp = os.path.join(OUT, 'scaler_stats.json'), os.path.join(OUT, 'feature_list.json')
stats, fl = json.load(open(sp)), json.load(open(fp))

def robust(col, years):
    vals = []
    for y in years:
        v = pq.read_table(os.path.join(DATA, f'feat_{y}.parquet'), columns=[col, 'drop_post_excl']).to_pandas()
        vals.append(v.loc[v['drop_post_excl'] == 0, col].to_numpy(dtype='float64'))
    v = np.concatenate(vals); q25, med, q75 = np.percentile(v, [25, 50, 75]); iqr = q75 - q25
    return {'median': float(med), 'iqr': float(iqr if iqr > 0 else 1.0), 'q25': float(q25), 'q75': float(q75), 'n': int(len(v))}

stats['cc_missing'] = robust('cc_missing', [y for y in cmsarm.YEARS if y >= 2017])
stats['year'] = robust('year', cmsarm.YEARS)
fl['primary_plus_year'] = fl['primary'] + ['year']
json.dump(stats, open(sp, 'w'), indent=1); json.dump(fl, open(fp, 'w'), indent=1)
print('cc_missing', stats['cc_missing']); print('year', stats['year'])
