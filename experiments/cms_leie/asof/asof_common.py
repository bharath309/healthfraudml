"""Protocol T (as-of evaluation) — shared paths and helpers. Lives in experiments/cms_leie/asof/; imports the arm's cmsarm.py
from the parent directory. All paths are the arm's: data/, out/, results/. Nothing here reads a LEIE file newer than 10 Sep 2026."""
import os, sys, json, hashlib, numpy as np, pandas as pd, pyarrow.parquet as pq
HERE=os.path.dirname(os.path.abspath(__file__)); ARM=os.environ.get('ASOF_ARM_DIR',os.path.dirname(HERE))
sys.path.insert(0,ARM)
import cmsarm                                           # noqa: E402  (the arm's loader: DATA, OUT, YEARS, features(), _load_rows)
DATA=cmsarm.DATA; OUT=cmsarm.OUT; RES=os.path.join(ARM,'results','asof'); os.makedirs(RES,exist_ok=True)
T_YEARS=[2018,2019,2020,2021,2022]
BUDGETS=[0.005,0.01,0.02,0.05]
SMOKE=os.environ.get('ASOF_SMOKE','0')=='1'
N_TREES={'gb':500,'rf':500} if not SMOKE else {'gb':20,'rf':20}
FEATS=cmsarm.features('primary')
NUMERIC=FEATS[:48]            # 37 published numerics + 11 derived ratios (feature_list order); indicators and categoricals follow
assert FEATS[48]=='block_suppressed' and FEATS[50]=='Rndrng_Prvdr_Type_code', FEATS[48:51]
PTYPE_COL='Rndrng_Prvdr_Type_code'; PAY_COL='Tot_Mdcr_Pymt_Amt'
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda: f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def known_table():
    p=os.path.join(DATA,'asof_known.parquet'); assert os.path.exists(p),'run asof_labels.py first'
    return pd.read_parquet(p)
def protocol_hash():
    p=os.path.join(HERE,'protocol_T_frozen.md'); return sha256(p) if os.path.exists(p) else None
def year_rows(y,cols):
    """all kept rows of year y with the requested feature columns (float32) + npi (int64)."""
    t=pq.read_table(os.path.join(DATA,f'feat_{y}.parquet'),columns=['Rndrng_NPI']+cols).to_pandas()
    t['npi']=t['Rndrng_NPI'].astype('int64'); return t.drop(columns=['Rndrng_NPI'])
def rank_order(score,npi):
    """ranking within a year: score descending, NPI ascending on ties (capacity_table.py convention)."""
    return np.lexsort((npi,-score))
def write_json(obj,path):
    json.dump(obj,open(path,'w'),indent=1,default=lambda o: float(o) if isinstance(o,(np.floating,)) else int(o) if isinstance(o,(np.integer,)) else str(o))
