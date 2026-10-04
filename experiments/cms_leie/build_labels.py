#!/usr/bin/env python3
"""Gate 1, step 2: LEIE-based fraud labels for every CMS provider-year, following the brief (§2.3):
- LEIE rows with a 10-digit NPI; EXCLTYPE normalised (lower-case, letters/digits only);
- primary authority set = 1128(a)(1), (a)(2), (a)(3), (b)(4), (b)(7), (c)(3)(G)(i), (c)(3)(G)(ii)  [ref 11, Table 3];
- exclusion year = year of min(EXCLDATE, WAIVERDATE if present), +1 if that date's month is 7-12  [refs 11, 16];
- is_fraud = 1 for provider-years with year < exclusion year of a matched NPI; provider-years with year >= exclusion
  year for matched NPIs are DROPPED (flag drop_post_excl = 1) and counted;
- secondary sets: is_fraud_mandatory (a1, a2, a3, c3gi, c3gii), is_fraud_5yr (primary, year >= exclusion year - 5).
Writes data/labels_<year>.parquet (Rndrng_NPI, year, Rndrng_Prvdr_Ent_Cd, Rndrng_Prvdr_Type, Rndrng_Prvdr_Cntry,
is_fraud, is_fraud_mandatory, is_fraud_5yr, drop_post_excl, excl_year) and out/label_report.{json,md}."""
import os, re, json, collections
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.csv as pacsv, pyarrow as pa
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'data'); OUT=os.path.join(HERE,'out')
YEARS=list(range(2013,2025))
PRIMARY={'1128a1','1128a2','1128a3','1128b4','1128b7','1128c3gi','1128c3gii'}
MANDATORY={'1128a1','1128a2','1128a3','1128c3gi','1128c3gii'}
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower()) if s is not None else ''
cols=['LASTNAME','FIRSTNAME','MIDNAME','BUSNAME','GENERAL','SPECIALTY','UPIN','NPI','DOB','ADDRESS','CITY','STATE','ZIP','EXCLTYPE','EXCLDATE','REINDATE','WAIVERDATE','WVRSTATE']
leie=pacsv.read_csv(os.path.join(DATA,'LEIE_UPDATED.csv'),convert_options=pacsv.ConvertOptions(column_types={c:pa.string() for c in cols})).to_pandas()
rep={'leie_rows':int(len(leie))}
leie['NPI']=leie['NPI'].fillna('').str.strip()
leie['npi_ok']=leie['NPI'].str.fullmatch(r'\d{10}') & (leie['NPI']!='0000000000')
rep['leie_rows_with_npi']=int(leie['npi_ok'].sum())
leie['excl_norm']=leie['EXCLTYPE'].map(norm)
rep['excltype_distinct']={k:int(v) for k,v in leie['EXCLTYPE'].fillna('<blank>').value_counts().items()}
rep['excltype_normalised_mapping']={k:sorted(set(leie.loc[leie['excl_norm']==k,'EXCLTYPE'].fillna('<blank>'))) for k in sorted(leie['excl_norm'].unique())}
leie['in_primary']=leie['excl_norm'].isin(PRIMARY); leie['in_mandatory']=leie['excl_norm'].isin(MANDATORY)
rep['leie_rows_primary_set']=int(leie['in_primary'].sum()); rep['leie_rows_primary_set_with_npi']=int((leie['in_primary']&leie['npi_ok']).sum())
rep['leie_rows_mandatory_set_with_npi']=int((leie['in_mandatory']&leie['npi_ok']).sum())
rep['leie_rows_with_reindate']=int(leie['REINDATE'].fillna('').str.strip().replace('00000000','').ne('').sum())
rep['leie_rows_with_waiverdate']=int(leie['WAIVERDATE'].fillna('').str.strip().replace('00000000','').ne('').sum())
def pdate(s):
    s=(s or '').strip()
    return pd.to_datetime(s,format='%Y%m%d',errors='coerce') if re.fullmatch(r'\d{8}',s) and s!='00000000' else pd.NaT
leie['excl_dt']=leie['EXCLDATE'].map(pdate); leie['waiver_dt']=leie['WAIVERDATE'].map(pdate)
rep['leie_rows_bad_excldate']=int(leie['excl_dt'].isna().sum())
end=leie['excl_dt'].where(leie['waiver_dt'].isna()|(leie['excl_dt']<=leie['waiver_dt']),leie['waiver_dt'])
leie['end_dt']=end
leie['excl_year']=(end.dt.year+(end.dt.month>=7).astype('Int64')).astype('Int64')
rep['excl_year_distribution_primary_with_npi']={int(k):int(v) for k,v in leie.loc[leie['in_primary']&leie['npi_ok'],'excl_year'].value_counts().sort_index().items()}
sel=leie[leie['npi_ok']&leie['excl_year'].notna()]
prim=sel[sel['in_primary']].groupby('NPI')['excl_year'].min().rename('excl_year_primary')
mand=sel[sel['in_mandatory']].groupby('NPI')['excl_year'].min().rename('excl_year_mandatory')
anyset=sel.groupby('NPI')['excl_year'].min().rename('excl_year_any')
npi_tbl=pd.concat([prim,mand,anyset],axis=1)
rep['distinct_npis_primary']=int(prim.shape[0]); rep['distinct_npis_mandatory']=int(mand.shape[0]); rep['distinct_npis_any_authority']=int(anyset.shape[0])
per_year={}; type_counts=collections.Counter(); ent_counts=collections.Counter(); matched_npis=set(); matched_any=set()
for y in YEARS:
    t=pq.read_table(os.path.join(DATA,f'by_provider_{y}.parquet'),columns=['Rndrng_NPI','Rndrng_Prvdr_Ent_Cd','Rndrng_Prvdr_Type','Rndrng_Prvdr_Cntry','Rndrng_Prvdr_State_Abrvtn']).to_pandas()
    n_all=len(t); t['year']=y
    assert t['Rndrng_NPI'].is_unique, (y,'NPI not unique')
    us=t['Rndrng_Prvdr_Cntry'].eq('US'); n_us=int(us.sum())
    t=t[us].merge(npi_tbl,left_on='Rndrng_NPI',right_index=True,how='left')
    ep=t['excl_year_primary']; em=t['excl_year_mandatory']
    t['drop_post_excl']=(ep.notna()&(t['year']>=ep)).astype('int8')
    t['is_fraud']=(ep.notna()&(t['year']<ep)).astype('int8')
    t['is_fraud_mandatory']=(em.notna()&(t['year']<em)).astype('int8')
    t['is_fraud_5yr']=(ep.notna()&(t['year']<ep)&(t['year']>=ep-5)).astype('int8')
    t['excl_year']=ep.astype('Int64')
    matched_npis|=set(t.loc[ep.notna(),'Rndrng_NPI']); matched_any|=set(t.loc[t['excl_year_any'].notna(),'Rndrng_NPI'])
    kept=t[t['drop_post_excl']==0]
    per_year[y]={'rows_file':n_all,'rows_us':n_us,'rows_dropped_post_exclusion':int(t['drop_post_excl'].sum()),'rows_kept':int(len(kept)),
                 'fraud_primary':int(kept['is_fraud'].sum()),'fraud_mandatory':int(kept['is_fraud_mandatory'].sum()),'fraud_5yr':int(kept['is_fraud_5yr'].sum()),
                 'fraud_rate_primary_pct':round(100*kept['is_fraud'].mean(),4),
                 'entity_I':int((kept['Rndrng_Prvdr_Ent_Cd']=='I').sum()),'entity_O':int((kept['Rndrng_Prvdr_Ent_Cd']=='O').sum()),
                 'fraud_entity_I':int(((kept['Rndrng_Prvdr_Ent_Cd']=='I')&(kept['is_fraud']==1)).sum()),'fraud_entity_O':int(((kept['Rndrng_Prvdr_Ent_Cd']=='O')&(kept['is_fraud']==1)).sum())}
    type_counts.update(kept.loc[kept['is_fraud']==1,'Rndrng_Prvdr_Type'].fillna('<blank>').tolist()); ent_counts.update(kept.loc[kept['is_fraud']==1,'Rndrng_Prvdr_Ent_Cd'].fillna('<blank>').tolist())
    t[['Rndrng_NPI','year','Rndrng_Prvdr_Ent_Cd','Rndrng_Prvdr_Type','Rndrng_Prvdr_Cntry','Rndrng_Prvdr_State_Abrvtn','is_fraud','is_fraud_mandatory','is_fraud_5yr','drop_post_excl','excl_year']].to_parquet(os.path.join(DATA,f'labels_{y}.parquet'),index=False)
    print(y,per_year[y],flush=True)
tot=lambda k: int(sum(v[k] for v in per_year.values()))
rep['per_year']=per_year
rep['totals']={'rows_us':tot('rows_us'),'rows_kept':tot('rows_kept'),'rows_dropped_post_exclusion':tot('rows_dropped_post_exclusion'),'fraud_primary':tot('fraud_primary'),'fraud_mandatory':tot('fraud_mandatory'),'fraud_5yr':tot('fraud_5yr'),
               'fraud_rate_primary_pct':round(100*tot('fraud_primary')/tot('rows_kept'),4),'matched_npis_primary_in_cms':len(matched_npis),'npis_with_any_authority_in_cms':len(matched_any)}
rep['fraud_rows_top_provider_types']=[[k,int(v)] for k,v in type_counts.most_common(15)]
rep['fraud_rows_entity_type']={k:int(v) for k,v in ent_counts.items()}
P=rep['totals']['fraud_primary']; rep['sampling_ratio_rule']={'P':P,'negatives_per_positive':20 if P>=2000 else (50 if P>=1000 else None),'rule':'20 if P>=2000; 50 if 1000<=P<2000; stop if P<1000 (brief §2, fixed before any result)'}
json.dump(rep,open(os.path.join(OUT,'label_report.json'),'w'),indent=1,default=int)
print(json.dumps({k:v for k,v in rep.items() if k not in ('per_year','excltype_normalised_mapping','excltype_distinct')},indent=1,default=int))
