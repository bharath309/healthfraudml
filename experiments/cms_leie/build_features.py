#!/usr/bin/env python3
"""Gate 2 build: per-year feature tables for the CMS-LEIE arm (brief §3.2), written BEFORE any model is trained.
Primary feature set = columns common to all twelve files (2013-2016 files have 56 columns, no chronic-condition block;
2017-2024 files have 81) -> 37 published numerics + 11 derived ratios + 2 suppression indicators (block_suppressed;
bene_demo_n_missing = number of suppressed demographic cells, 0-15) + 5 label-encoded categoricals = 55 features. The 25 chronic-condition percentages (2017+ only) are stored too, for the pre-specified
sensitivity on 2017-2024 (+ cc_missing indicator). Blanks -> 0 with an indicator; Bene_Avg_Risk_Scre blank -> year median
+ risk_missing. Label encoders are fitted on the union of categories over all years (sorted, as sklearn's LabelEncoder).
RobustScaler statistics (median, IQR; IQR 0 -> 1, as sklearn) are computed over ALL kept rows of all years and saved to
out/scaler_stats.json; scaling is applied at load time. Outputs: data/feat_<year>.parquet, out/feature_list.json."""
import os, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'data'); OUT=os.path.join(HERE,'out')
YEARS=list(range(2013,2025))
NUM=['Tot_HCPCS_Cds','Tot_Benes','Tot_Srvcs','Tot_Sbmtd_Chrg','Tot_Mdcr_Alowd_Amt','Tot_Mdcr_Pymt_Amt','Tot_Mdcr_Stdzd_Amt',
     'Drug_Tot_HCPCS_Cds','Drug_Tot_Benes','Drug_Tot_Srvcs','Drug_Sbmtd_Chrg','Drug_Mdcr_Alowd_Amt','Drug_Mdcr_Pymt_Amt','Drug_Mdcr_Stdzd_Amt',
     'Med_Tot_HCPCS_Cds','Med_Tot_Benes','Med_Tot_Srvcs','Med_Sbmtd_Chrg','Med_Mdcr_Alowd_Amt','Med_Mdcr_Pymt_Amt','Med_Mdcr_Stdzd_Amt',
     'Bene_Avg_Age','Bene_Age_LT_65_Cnt','Bene_Age_65_74_Cnt','Bene_Age_75_84_Cnt','Bene_Age_GT_84_Cnt','Bene_Feml_Cnt','Bene_Male_Cnt',
     'Bene_Race_Wht_Cnt','Bene_Race_Black_Cnt','Bene_Race_API_Cnt','Bene_Race_Hspnc_Cnt','Bene_Race_NatInd_Cnt','Bene_Race_Othr_Cnt',
     'Bene_Dual_Cnt','Bene_Ndual_Cnt','Bene_Avg_Risk_Scre']
BENE_CNT=['Bene_Avg_Age','Bene_Age_LT_65_Cnt','Bene_Age_65_74_Cnt','Bene_Age_75_84_Cnt','Bene_Age_GT_84_Cnt','Bene_Feml_Cnt','Bene_Male_Cnt',
          'Bene_Race_Wht_Cnt','Bene_Race_Black_Cnt','Bene_Race_API_Cnt','Bene_Race_Hspnc_Cnt','Bene_Race_NatInd_Cnt','Bene_Race_Othr_Cnt','Bene_Dual_Cnt','Bene_Ndual_Cnt']
CAT=['Rndrng_Prvdr_Type','Rndrng_Prvdr_State_Abrvtn','Rndrng_Prvdr_Ent_Cd','Rndrng_Prvdr_RUCA','Rndrng_Prvdr_Mdcr_Prtcptg_Ind']
CC=[c for c in pq.read_schema(os.path.join(DATA,'by_provider_2024.parquet')).names if c.startswith('Bene_CC_')]
assert len(CC)==25,len(CC)
DERIVED=['srvcs_per_bene','chrg_per_srvc','alowd_per_srvc','pymt_over_alowd','stdzd_over_pymt','drug_share_pymt','hcpcs_per_1000_srvcs','dual_share','female_share','share_65_74','share_85p']
IND=['block_suppressed','bene_demo_n_missing']   # Drug/Med suppression flags coincide row-wise; risk score is never blank; 'any demographic missing' is ~97% of rows, so the count of suppressed demographic cells is used instead
def ruca1(s): return s.fillna('').astype(str).str.strip().str[:1].replace('', '<blank>')
# pass 1: category vocabularies over all years
vocab={c:set() for c in CAT}
for y in YEARS:
    t=pq.read_table(os.path.join(DATA,f'by_provider_{y}.parquet'),columns=CAT).to_pandas()
    for c in CAT:
        s=ruca1(t[c]) if c=='Rndrng_Prvdr_RUCA' else t[c].fillna('<blank>').astype(str).str.strip().replace('','<blank>')
        vocab[c]|=set(s.unique())
codes={c:{v:i for i,v in enumerate(sorted(vocab[c]))} for c in CAT}
json.dump({c:sorted(vocab[c]) for c in CAT},open(os.path.join(OUT,'category_vocab.json'),'w'),indent=1)
def safe_div(a,b): return np.where(b>0,a/np.where(b>0,b,1),0.0)
feats=NUM+DERIVED+IND+[c+'_code' for c in CAT]
for y in YEARS:
    cols=['Rndrng_NPI']+NUM+['Drug_Sprsn_Ind','Med_Sprsn_Ind']+CAT+([c for c in CC] if y>=2017 else [])
    t=pq.read_table(os.path.join(DATA,f'by_provider_{y}.parquet'),columns=cols).to_pandas()
    lab=pd.read_parquet(os.path.join(DATA,f'labels_{y}.parquet'))
    t=lab[['Rndrng_NPI','year','is_fraud','is_fraud_mandatory','is_fraud_5yr','drop_post_excl']].merge(t,on='Rndrng_NPI',how='left')
    assert len(t)==len(lab)
    X=pd.DataFrame(index=t.index)
    num=t[NUM].apply(pd.to_numeric,errors='coerce')
    X['block_suppressed']=t['Drug_Sprsn_Ind'].fillna('').str.strip().ne('').astype('float32')
    X['bene_demo_n_missing']=num[BENE_CNT].isna().sum(axis=1).astype('float32')
    risk_missing=int(num['Bene_Avg_Risk_Scre'].isna().sum())
    med_risk=float(num['Bene_Avg_Risk_Scre'].median())
    num['Bene_Avg_Risk_Scre']=num['Bene_Avg_Risk_Scre'].fillna(med_risk)
    num=num.fillna(0.0)
    for c in NUM: X[c]=num[c].astype('float32')
    X['srvcs_per_bene']=safe_div(num['Tot_Srvcs'],num['Tot_Benes']); X['chrg_per_srvc']=safe_div(num['Tot_Sbmtd_Chrg'],num['Tot_Srvcs'])
    X['alowd_per_srvc']=safe_div(num['Tot_Mdcr_Alowd_Amt'],num['Tot_Srvcs']); X['pymt_over_alowd']=safe_div(num['Tot_Mdcr_Pymt_Amt'],num['Tot_Mdcr_Alowd_Amt'])
    X['stdzd_over_pymt']=safe_div(num['Tot_Mdcr_Stdzd_Amt'],num['Tot_Mdcr_Pymt_Amt']); X['drug_share_pymt']=safe_div(num['Drug_Mdcr_Pymt_Amt'],num['Tot_Mdcr_Pymt_Amt'])
    X['hcpcs_per_1000_srvcs']=1000*safe_div(num['Tot_HCPCS_Cds'],num['Tot_Srvcs']); X['dual_share']=safe_div(num['Bene_Dual_Cnt'],num['Bene_Dual_Cnt']+num['Bene_Ndual_Cnt'])
    X['female_share']=safe_div(num['Bene_Feml_Cnt'],num['Bene_Feml_Cnt']+num['Bene_Male_Cnt']); X['share_65_74']=safe_div(num['Bene_Age_65_74_Cnt'],num['Tot_Benes']); X['share_85p']=safe_div(num['Bene_Age_GT_84_Cnt'],num['Tot_Benes'])
    for c in CAT:
        s=ruca1(t[c]) if c=='Rndrng_Prvdr_RUCA' else t[c].fillna('<blank>').astype(str).str.strip().replace('','<blank>')
        X[c+'_code']=s.map(codes[c]).astype('float32')
    if y>=2017:
        cc=t[CC].apply(pd.to_numeric,errors='coerce'); X['cc_missing']=cc.isna().any(axis=1).astype('float32')
        for c in CC: X[c]=cc[c].fillna(0.0).astype('float32')
    else:
        X['cc_missing']=np.float32(1.0)
        for c in CC: X[c]=np.float32(np.nan)
    X=X[feats+['cc_missing']+CC].astype('float32')
    out=pd.concat([t[['Rndrng_NPI','year','is_fraud','is_fraud_mandatory','is_fraud_5yr','drop_post_excl']].reset_index(drop=True),X.reset_index(drop=True)],axis=1)
    out.to_parquet(os.path.join(DATA,f'feat_{y}.parquet'),index=False)
    print(y,len(out),'rows',len(feats),'primary features; risk median',round(med_risk,3),'| block suppressed',int(X['block_suppressed'].sum()),'| mean demo cells missing',round(float(X['bene_demo_n_missing'].mean()),2),'| risk missing',risk_missing,flush=True)
json.dump({'primary':feats,'sensitivity_cc_2017plus':feats+['cc_missing']+CC,'numeric_published':NUM,'derived':DERIVED,'indicators':IND,'categorical_label_encoded':CAT},open(os.path.join(OUT,'feature_list.json'),'w'),indent=1)
# RobustScaler statistics over all kept rows, all years (primary + CC columns; CC over 2017+ only)
stats={}
allcols=feats+CC
for c in allcols:
    vals=[]
    for y in YEARS:
        if c in CC and y<2017: continue
        v=pq.read_table(os.path.join(DATA,f'feat_{y}.parquet'),columns=[c,'drop_post_excl']).to_pandas()
        vals.append(v.loc[v['drop_post_excl']==0,c].to_numpy(dtype='float64'))
    v=np.concatenate(vals); q25,med,q75=np.percentile(v,[25,50,75]); iqr=q75-q25
    stats[c]={'median':float(med),'iqr':float(iqr if iqr>0 else 1.0),'q25':float(q25),'q75':float(q75),'n':int(len(v))}
json.dump(stats,open(os.path.join(OUT,'scaler_stats.json'),'w'),indent=1)
print('scaler stats written for',len(stats),'columns')
