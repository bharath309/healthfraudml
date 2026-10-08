#!/usr/bin/env python3
"""Protocol T step 1: as-of labels. From the LEIE file of 10 Sep 2026 (data/LEIE_UPDATED.csv) and the arm's labels_<year>.parquet,
write data/asof_known.parquet with, per kept row: npi, year, is_fraud, excl_year, excl_end_date (earliest fraud-authority
end date = min(EXCLDATE, WAIVERDATE) per record, as build_labels.py), is_fraud_3yr_<t> and known_<t> for t in T_YEARS.
Label-only; identical on both machines. Writes results/asof/labels_report.json. No model is trained here."""
import os, re, json, numpy as np, pandas as pd, pyarrow as pa, pyarrow.csv as pacsv, pyarrow.parquet as pq
from asof_common import DATA, RES, T_YEARS, write_json, sha256
cols=['LASTNAME','FIRSTNAME','MIDNAME','BUSNAME','GENERAL','SPECIALTY','UPIN','NPI','DOB','ADDRESS','CITY','STATE','ZIP','EXCLTYPE','EXCLDATE','REINDATE','WAIVERDATE','WVRSTATE']
leie_path=os.path.join(DATA,'LEIE_UPDATED.csv')
leie=pacsv.read_csv(leie_path,convert_options=pacsv.ConvertOptions(column_types={c:pa.string() for c in cols})).to_pandas()
norm=lambda s: re.sub(r'[^a-z0-9]','',str(s).lower()); leie['code']=leie['EXCLTYPE'].map(norm)
FRAUD={'1128a1','1128a2','1128a3','1128b4','1128b7'}
npi=leie['NPI'].fillna('').str.strip(); ok=npi.str.fullmatch(r'\d{10}')&(npi!='0000000000')
def pdate(s):
    s=(s or '').strip()
    return pd.to_datetime(s,format='%Y%m%d',errors='coerce') if re.fullmatch(r'\d{8}',s) and s!='00000000' else pd.NaT
leie['excl_dt']=leie['EXCLDATE'].map(pdate); leie['waiver_dt']=leie['WAIVERDATE'].map(pdate)
end=leie['excl_dt'].where(leie['waiver_dt'].isna()|(leie['excl_dt']<=leie['waiver_dt']),leie['waiver_dt']); leie['end_dt']=end
sel=leie[ok&leie['code'].isin(FRAUD)&leie['end_dt'].notna()].copy(); sel['npi']=sel['NPI'].astype('int64')
first=sel.groupby('npi')['end_dt'].min().rename('excl_end_date')
parts=[]
for y in range(2013,2025):
    t=pq.read_table(os.path.join(DATA,f'labels_{y}.parquet'),columns=['Rndrng_NPI','year','is_fraud','excl_year','drop_post_excl']).to_pandas()
    t=t[t['drop_post_excl']==0].copy(); t['npi']=t['Rndrng_NPI'].astype('int64'); parts.append(t[['npi','year','is_fraud','excl_year']])
K=pd.concat(parts,ignore_index=True); K['year']=K['year'].astype('int16'); K['is_fraud']=K['is_fraud'].astype('int8')
K=K.merge(first,on='npi',how='left')
# consistency with the arm: every positive row has an end date; excl_year of the arm equals year(end)+(month>=7) for positives
pos=K['is_fraud']==1; assert K.loc[pos,'excl_end_date'].notna().all(),'positive row without an LEIE end date'
ey=(K.loc[pos,'excl_end_date'].dt.year+(K.loc[pos,'excl_end_date'].dt.month>=7).astype(int)); mism=int((ey.values!=K.loc[pos,'excl_year'].astype(int).values).sum())
rep={'leie_file':os.path.basename(leie_path),'leie_sha256':sha256(leie_path),'rows_kept':int(len(K)),'positives':int(pos.sum()),'npis_with_end_date':int(first.shape[0]),'excl_year_mismatches_vs_arm':mism}
assert mism==0, f'excl_year mismatch vs the arm for {mism} positive rows'
for t in T_YEARS:
    K[f'known_{t}']=(pos&(K['excl_end_date']<pd.Timestamp(year=t,month=1,day=1))).astype('int8')
    K[f'is_fraud_3yr_{t}']=(pos&(K['excl_year']<=t+3)).astype('int8')
    tr=K['year']<=t-1; te=K['year']==t
    rep[str(t)]={'train_rows':int(tr.sum()),'train_eventual_positives':int((tr&pos).sum()),'train_known_positives':int(K.loc[tr,f'known_{t}'].sum()),
                 'share_known':float(K.loc[tr,f'known_{t}'].sum()/max(1,(tr&pos).sum())),'test_rows':int(te.sum()),'test_positives':int((te&pos).sum()),'test_positives_3yr':int(K.loc[te,f'is_fraud_3yr_{t}'].sum())}
out=os.path.join(DATA,'asof_known.parquet'); K.to_parquet(out,index=False); rep['asof_known_sha256']=sha256(out)
write_json(rep,os.path.join(RES,'labels_report.json')); print(json.dumps(rep,indent=1,default=str))
