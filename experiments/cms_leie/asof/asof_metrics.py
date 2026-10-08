#!/usr/bin/env python3
"""Protocol T step 4: metrics, hindsight comparison, triage and allocation comparisons, bootstrap intervals, decision rules T1-T4.
Reads results/asof/scores_<t>.parquet (+ scores_prev_<t>.parquet for the allocation), the arm's results/oof_fold<k>.parquet,
data/asof_known.parquet. Writes results/asof/summary.json and summary.md. Every number in the paper comes from summary.json."""
import os, glob, json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from asof_common import ARM, RES, T_YEARS, BUDGETS, known_table, rank_order, write_json, protocol_hash
B=int(os.environ.get('ASOF_BOOT','1000')); VARS=['gb_A','rf_A','gb_B','rf_B']
def topk_metrics(score,y,npi,pay,budgets):
    o=rank_order(score,npi); ys=y[o]; ps=pay[o]; n=len(y); P=ys.sum(); D=ps[ys==1].sum(); prev=P/n; out={}
    for k in budgets:
        m=int(np.floor(k*n)); tp=int(ys[:m].sum()); out[str(k)]={'rows':m,'tp':tp,'recall':tp/P,'precision':tp/m,'lift':(tp/m)/prev,'dollar_recall':float(ps[:m][ys[:m]==1].sum()/D),'distinct_npis':int(len(np.unique(npi[o][:m][ys[:m]==1])))}
    out['auc_pr']=float(average_precision_score(y,score)); out['auc_roc']=float(roc_auc_score(y,score)); out['prevalence']=float(prev); out['n']=int(n); out['positives']=int(P); return out
def recall_at(score,y,npi,k,w=None):
    """recall@k with optional bootstrap weights w (providers resampled with replacement): prefix of the fixed ranking until floor(k*sum w)."""
    o=rank_order(score,npi); ys=y[o]
    if w is None: m=int(np.floor(k*len(y))); return ys[:m].sum()/ys.sum()
    ws=w[o]; cw=np.cumsum(ws); m=np.searchsorted(cw,np.floor(k*cw[-1]),side='right'); return float((ws[:m]*ys[:m]).sum()/(ws*ys).sum())
def alloc_recall(S,P,k,w=None):
    """yield-aware allocation at budget k using year t-1 flags by the model built at t (P: prev-year frame with p, known_t, ptype)."""
    kprev=int(np.floor(k*len(P))); o=rank_order(P['p'].values,P['npi'].values)[:kprev]; flagged=P.iloc[o]
    yl=flagged.groupby('ptype')['known'].agg(['sum','size']); pooled=flagged['known'].mean()
    yield_s={int(s):(r['sum']/r['size'] if r['size']>=50 else pooled) for s,r in yl.iterrows()}
    n=len(S); budget=int(np.floor(k*n)); types=S['ptype'].values; ns=pd.Series(types).value_counts()
    weight={int(s):ns[s]*yield_s.get(int(s),pooled) for s in ns.index}; tot=sum(weight.values())
    y=S['is_fraud'].values; p=S['p'].values; npi=S['npi'].values; ww=np.ones(n) if w is None else w
    if tot<=0: return recall_at(p,y,npi,k,w)      # degenerate case (every yield estimate 0): the allocation is the global ranking (protocol §6)
    sel=np.zeros(n,dtype=bool); used=0
    for s,wt in weight.items():
        bs=int(np.floor(budget*wt/tot)); idx=np.where(types==s)[0]
        if bs==0 or len(idx)==0: continue
        o=idx[rank_order(p[idx],npi[idx])]
        if w is None: take=o[:min(bs,len(o))]
        else: cw=np.cumsum(ww[o]); take=o[:np.searchsorted(cw,min(bs,cw[-1]),side='right')]
        sel[take]=True; used+=len(take)
    rest=budget-used
    if rest>0:   # unused budget returns to the global ranking
        o=rank_order(p,npi); o=o[~sel[o]]
        if w is None: sel[o[:rest]]=True
        else: cw=np.cumsum(ww[o]); sel[o[:np.searchsorted(cw,rest,side='right')]]=True
    return float((ww*y*sel).sum()/(ww*y).sum())
K=known_table(); summary={'protocol_sha256':protocol_hash(),'B':B,'years':{}}
oof=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ARM,'results','oof_fold*.parquet')))],ignore_index=True) if glob.glob(os.path.join(ARM,'results','oof_fold*.parquet')) else None
diffs={'T2':[], 'T3':[], 'T4':[]}; boot={'T2':np.zeros(B),'T4':np.zeros(B)}
for t in T_YEARS:
    S=pd.read_parquet(os.path.join(RES,f'scores_{t}.parquet')); y=S['is_fraud'].values.astype(int); y3=S['is_fraud_3yr'].values.astype(int); npi=S['npi'].values; pay=S['pay'].values
    Y={'asof':{},'asof_3yr':{}}
    for v in VARS:
        if f'p_{v}' in S: Y['asof'][v]=topk_metrics(S[f'p_{v}'].values.astype('float64'),y,npi,pay,BUDGETS); Y['asof_3yr'][v]=topk_metrics(S[f'p_{v}'].values.astype('float64'),y3,npi,pay,BUDGETS)
    if oof is not None:
        H=oof[oof['year']==t].merge(S[['npi','pay']],on='npi',how='inner'); Y['hindsight']={}
        for m in ['gb','rf']:
            if f'p_{m}' in H: Y['hindsight'][m]=topk_metrics(H[f'p_{m}'].values.astype('float64'),H['is_fraud'].values.astype(int),H['npi'].values,H['pay'].values,BUDGETS)
        if 'gb_A' in Y['asof'] and 'gb' in Y['hindsight']:
            la=Y['asof']['gb_A']['0.01']['lift']; Y['hindsight_optimism_ratio_gb']=(Y['hindsight']['gb']['0.01']['lift']/la) if la>0 else None   # undefined when the as-of lift is 0; T1 compares the lifts directly
    kt=K[K['year']<=t-1]; Y['share_of_training_positives_known_at_t']=float(kt[f'known_{t}'].sum()/max(1,kt['is_fraud'].sum()))   # label completeness of the training years
    # triage rules on GB-A at 1%
    if 'p_gb_A' in S:
        p=S['p_gb_A'].values.astype('float64'); Y['triage_1pct']={}
        for name,sc in [('p',p),('p_x_payment',p*pay),('payment_only',pay)]:
            Y['triage_1pct'][name]=topk_metrics(sc,y,npi,pay,[0.01])['0.01']
        diffs['T3'].append(Y['triage_1pct']['p_x_payment']['dollar_recall']-Y['triage_1pct']['p']['dollar_recall'])
    # T2 paired difference and bootstrap (providers = rows within year t)
    rng=np.random.default_rng(2026+t); W=rng.multinomial(len(y),np.full(len(y),1/len(y)),size=B).astype('float64') if B>0 else None
    if 'p_gb_A' in S and 'p_gb_B' in S:
        a=S['p_gb_A'].values.astype('float64'); b=S['p_gb_B'].values.astype('float64'); d=recall_at(b,y,npi,0.01)-recall_at(a,y,npi,0.01); diffs['T2'].append(d)
        for i in range(B): boot['T2'][i]+=(recall_at(b,y,npi,0.01,W[i])-recall_at(a,y,npi,0.01,W[i]))/len(T_YEARS)
    # T4 allocation vs global (GB-A), needs prev-year scores by the model built at t
    pp=os.path.join(RES,f'scores_prev_{t}.parquet')
    if 'p_gb_A' in S and os.path.exists(pp):
        P=pd.read_parquet(pp).rename(columns={'p_gb_A':'p'}); P=P.merge(K[K['year']==t-1][['npi',f'known_{t}']].rename(columns={f'known_{t}':'known'}),on='npi',how='left').fillna({'known':0})
        Sg=S.rename(columns={'p_gb_A':'p'}); g=recall_at(Sg['p'].values.astype('float64'),y,npi,0.01); al=alloc_recall(Sg,P,0.01); Y['allocation_1pct']={'global_recall':g,'yield_aware_recall':al}; diffs['T4'].append(al-g)
        for i in range(B): boot['T4'][i]+=(alloc_recall(Sg,P,0.01,W[i])-recall_at(Sg['p'].values.astype('float64'),y,npi,0.01,W[i]))/len(T_YEARS)
    summary['years'][str(t)]=Y; print('year',t,'done',flush=True)
def ci(a): return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]
R={}
yrs=[summary['years'][str(t)] for t in T_YEARS]
if all('hindsight_optimism_ratio_gb' in Y for Y in yrs):
    wins=sum(Y['hindsight']['gb']['0.01']['lift']>Y['asof']['gb_A']['0.01']['lift'] for Y in yrs); R['T1']={'rule':'as-of lift@1% < hindsight lift@1% in >=4 of 5 years','years_met':int(wins),'met':bool(wins>=4),'ratios':[Y.get('hindsight_optimism_ratio_gb') for Y in yrs]}
if len(diffs['T2'])==len(T_YEARS):
    wins=sum(d>0 for d in diffs['T2']); c=ci(boot['T2']); R['T2']={'rule':'GB-B > GB-A recall@1% in >=4 of 5 years and bootstrap 95% CI of the 5-year mean difference excludes 0','years_met':wins,'mean_diff':float(np.mean(diffs['T2'])),'ci':c,'met':bool(wins>=4 and (c[0]>0 or c[1]<0))}
if len(diffs['T3'])==len(T_YEARS):
    wins=sum(d>0 for d in diffs['T3']); R['T3']={'rule':'p x payment > p on dollar-weighted recall@1% in 5 of 5 years','years_met':wins,'diffs':diffs['T3'],'met':wins==5}
if len(diffs['T4'])==len(T_YEARS):
    wins=sum(d>0 for d in diffs['T4']); c=ci(boot['T4']); R['T4']={'rule':'yield-aware > global recall@1% in >=4 of 5 years and bootstrap 95% CI excludes 0','years_met':wins,'mean_diff':float(np.mean(diffs['T4'])),'ci':c,'met':bool(wins>=4 and (c[0]>0 or c[1]<0))}
summary['decision_rules']=R; write_json(summary,os.path.join(RES,'summary.json'))
L=['# Protocol T — summary (generated by asof_metrics.py; every number here is in summary.json)',f'protocol sha256: {summary["protocol_sha256"]}; bootstrap B = {B}','', '| t | known share | GB-A lift@1% | hindsight GB lift@1% | ratio | GB-A recall@1% | GB-B recall@1% | $-recall p | $-recall p×pay | alloc recall | global recall |','|---|---|---|---|---|---|---|---|---|---|---|']
for t in T_YEARS:
    Y=summary['years'][str(t)]; g=Y['asof'].get('gb_A',{}).get('0.01',{}); h=Y.get('hindsight',{}).get('gb',{}).get('0.01',{}); b=Y['asof'].get('gb_B',{}).get('0.01',{}); tr=Y.get('triage_1pct',{}); al=Y.get('allocation_1pct',{})
    f=lambda d,k,fmt='{:.3f}': fmt.format(d[k]) if k in d else '—'
    L.append(f"| {t} | {Y['share_of_training_positives_known_at_t']:.3f} | {f(g,'lift','{:.1f}')} | {f(h,'lift','{:.1f}')} | {('{:.2f}'.format(Y['hindsight_optimism_ratio_gb']) if Y.get('hindsight_optimism_ratio_gb') is not None else '—')} | {f(g,'recall')} | {f(b,'recall')} | {f(tr.get('p',{}),'dollar_recall')} | {f(tr.get('p_x_payment',{}),'dollar_recall')} | {f(al,'yield_aware_recall')} | {f(al,'global_recall')} |")
L+=['','## Decision rules']+[f"- {k}: {v['rule']} — years met {v['years_met']}/{len(T_YEARS)}"+(f"; mean diff {v['mean_diff']:.4f}, 95% CI {v['ci'][0]:.4f} to {v['ci'][1]:.4f}" if 'ci' in v else '')+f" → **{'MET' if v['met'] else 'NOT MET'}**" for k,v in R.items()]
open(os.path.join(RES,'summary.md'),'w').write('\n'.join(L)+'\n'); print('\n'.join(L))
