"""Shared loader for the CMS-LEIE arm: scaled feature matrices for a fold's pre-sampled training set and complete test fold.
Scaling = RobustScaler convention with statistics from out/scaler_stats.json (median/IQR over all kept rows, all years)."""
import os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'data'); OUT=os.path.join(HERE,'out')
YEARS=list(range(2013,2025))
FL=json.load(open(os.path.join(OUT,'feature_list.json'))); STATS=json.load(open(os.path.join(OUT,'scaler_stats.json')))
# Feature-file variants (5 Oct 2026): a set name ending in '_imp' reads data/feat_<year>_imp.parquet and out/scaler_stats_imp.json
# (suppressed demographic counts imputed with 5.5); every other set reads the frozen files. Feature NAMES are identical.
def _suffix(setname): return '_imp' if setname.endswith('_imp') else ''
_STATS_CACHE={'':STATS}
def _stats(setname):
    s=_suffix(setname)
    if s not in _STATS_CACHE: _STATS_CACHE[s]=json.load(open(os.path.join(OUT,f'scaler_stats{s}.json')))
    return _STATS_CACHE[s]
def features(setname='primary'): return FL[setname]
def _yrs(setname): return [y for y in YEARS if (y>=2017 or not setname.startswith('sensitivity_cc'))]   # only the chronic-condition set is restricted to 2017+
def train_file(k,ratio=20): return os.path.join(DATA,f'train_{k}.parquet' if int(ratio)==20 else f'train_{k}_r{int(ratio)}.parquet')
def _scale(X,feats,setname='primary'):
    st=_stats(setname); med=np.array([st[c]['median'] for c in feats],dtype='float64'); iqr=np.array([st[c]['iqr'] for c in feats],dtype='float64')
    return ((X.astype('float64')-med)/iqr)
def _load_rows(keys,feats,label='is_fraud',years=YEARS,setname='primary'):
    """keys: DataFrame(npi:int64, year:int16) -> X, y, npi, year in the order of keys (inner-joined per year)."""
    outX=[];outy=[];outn=[];outyr=[]; s=_suffix(setname)
    for y in years:
        kk=keys[keys['year']==y]
        if len(kk)==0: continue
        t=pq.read_table(os.path.join(DATA,f'feat_{y}{s}.parquet'),columns=['Rndrng_NPI',label]+feats).to_pandas()
        t['npi']=t['Rndrng_NPI'].astype('int64'); t=t.set_index('npi').loc[kk['npi'].values]
        outX.append(t[feats].to_numpy(dtype='float32')); outy.append(t[label].to_numpy(dtype='int8')); outn.append(t.index.to_numpy()); outyr.append(np.full(len(t),y,dtype='int16'))
    return np.vstack(outX),np.concatenate(outy),np.concatenate(outn),np.concatenate(outyr)
def load_train(k,setname='primary',label='is_fraud',ratio=20):
    feats=features(setname); keys=pd.read_parquet(train_file(k,ratio)); yrs=_yrs(setname)
    X,y,n,yr=_load_rows(keys[keys['year'].isin(yrs)],feats,label,yrs,setname); return _scale(X,feats,setname),y,n,yr
def load_test(k,setname='primary',label='is_fraud'):
    feats=features(setname); f=pd.read_parquet(os.path.join(DATA,'folds.parquet')); keys=f[f['fold']==k][['npi','year']]; yrs=_yrs(setname)
    X,y,n,yr=_load_rows(keys[keys['year'].isin(yrs)],feats,label,yrs,setname); return _scale(X,feats,setname),y,n,yr
