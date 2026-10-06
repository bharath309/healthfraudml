#!/usr/bin/env python3
"""Provider-level bootstrap intervals for the CMS-LEIE arm, from saved outputs only (no model is trained).
Frozen definition (5 Oct 2026):
- unit of resampling = provider (NPI), within each test fold: the providers of fold k are drawn with replacement (as many
  draws as there are providers in the fold); every row of a drawn provider enters with the provider's draw count as its
  weight; fold k uses numpy default_rng(2026 + k), one draw per replicate, shared by all models (paired);
- per replicate b (B = 1000) the per-fold weighted F1 (the arm's fixed predict rule: p > 0.5; SVC decision > 0), weighted
  AUC-PR (average precision, step-wise as scikit-learn) and weighted AUC-ROC (tie-aware, as scikit-learn) are computed and
  averaged over the five folds, exactly as summary.json averages per-fold metrics;
- interval = 2.5th and 97.5th percentiles of the B replicate means; paired differences use the same draws;
- models: gb, rf, mlp, nb, svm, ens (ensemble = member mean saved by summarise.py), gb200, rf200.
The weighted metrics are validated against scikit-learn (sample_weight) at start-up on a 200,000-row subsample with unit
and random integer weights (max |diff| must be < 1e-9, nan-safe). Resumable: replicates are stored every 50.
Usage: bootstrap_ci.py [--set primary] [--label is_fraud] [--ratio 20] [--B 1000] [--budget 540]"""
import os, sys, time, json, argparse, numpy as np, pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('--set',default='primary'); ap.add_argument('--label',default='is_fraud'); ap.add_argument('--ratio',type=int,default=20); ap.add_argument('--B',type=int,default=1000); ap.add_argument('--budget',type=float,default=540); ap.add_argument('--models',default='gb,rf,mlp,nb,svm,ens,gb200,rf200'); a=ap.parse_args()
T0=time.time(); HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,'data'); OOF=os.path.join(HERE,'results','oof')
tag=f'{a.set}_{a.label}'+('' if a.ratio==20 else f'_r{a.ratio}'); RES=os.path.join(HERE,'results','bootstrap'); os.makedirs(RES,exist_ok=True)
MODELS=a.models.split(','); METRICS=['f1','auc_pr','auc_roc']; SEED=2026
class ModelFold:
    """Pre-sorted structures so that one replicate costs one gather + one cumulative sum over the negatives."""
    def __init__(self,p,yhat,y,prov_index):
        self.pos=np.flatnonzero(y==1); self.neg=np.flatnonzero(y==0)
        self.tp=self.pos[yhat[self.pos]==1]; self.fn=self.pos[yhat[self.pos]==0]; self.fp=self.neg[yhat[self.neg]==1]
        ps=p[self.pos]; o=np.argsort(-ps,kind='stable'); self.pos_sorted=self.pos[o]; ps=ps[o]                     # positives, score desc
        self.pos_grp_starts=np.r_[0,np.flatnonzero(np.diff(ps))+1]                                                # tie groups among positives
        pn=p[self.neg]; on=np.argsort(-pn,kind='stable'); self.neg_prov_sorted=prov_index[self.neg[on]].astype('int32'); pn=pn[on]   # negatives, score desc
        neg_key=-pn                                                                                                # ascending
        self.a=np.searchsorted(neg_key,-ps,side='left'); self.b=np.searchsorted(neg_key,-ps,side='right')          # per positive: # negatives with score > s, >= s
        self.b_grp=self.b[self.pos_grp_starts]                                                                      # per positive tie group (same score -> same b)
    def metrics(self,cnt):
        """cnt: draw count per provider (float64). Returns f1, average precision, auc_roc under row weights cnt[prov]."""
        wtp=cnt[self.prov_tp].sum(); wfp=cnt[self.prov_fp].sum(); wfn=cnt[self.prov_fn].sum()
        f1=2*wtp/(2*wtp+wfp+wfn) if (2*wtp+wfp+wfn)>0 else 0.0
        wn=cnt[self.neg_prov_sorted]; C=np.r_[0.0,np.cumsum(wn)]; Wneg=C[-1]
        wp=cnt[self.prov_pos_sorted]; Wpos=wp.sum()
        if Wpos==0 or Wneg==0: return f1,float('nan'),float('nan')
        below=Wneg-C[self.b]; tie=C[self.b]-C[self.a]
        auc=float((wp*(below+0.5*tie)).sum()/(Wpos*Wneg))
        g=np.add.reduceat(wp,self.pos_grp_starts); ctp=np.cumsum(g); cfp=C[self.b_grp]; v=g>0      # groups with zero drawn weight contribute 0 (avoids 0/0)
        ap_=float(((g[v]/Wpos)*(ctp[v]/(ctp[v]+cfp[v]))).sum())
        return f1,ap_,auc
class Fold:
    def __init__(self,k):
        self.k=k; keys=pd.read_parquet(os.path.join(OOF,f'keys_fold{k}.parquet')); self.y=np.load(os.path.join(DATA,f'test_{a.set}_{k}_y_{a.label}.npy')).astype('int8')
        npi=keys['npi'].values.astype('int64'); _,self.prov_index=np.unique(npi,return_inverse=True); self.n_prov=int(self.prov_index.max())+1; self.n=len(self.y); self.models={}
        for m in MODELS:
            p=np.load(os.path.join(OOF,f'{tag}_{m}_fold{k}.npy')).astype('float64')
            yhat=(np.load(os.path.join(OOF,f'{tag}_{m}_fold{k}_decision.npy'))>0).astype('int8') if m in ('svm','svc200') else (p>0.5).astype('int8')
            mf=ModelFold(p,yhat,self.y,self.prov_index); mf.prov_tp=self.prov_index[mf.tp]; mf.prov_fp=self.prov_index[mf.fp]; mf.prov_fn=self.prov_index[mf.fn]; mf.prov_pos_sorted=self.prov_index[mf.pos_sorted]
            self.models[m]=mf
def validate(fold):
    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
    rng=np.random.default_rng(0); idx=np.sort(rng.choice(fold.n,size=min(200000,fold.n),replace=False)); y=fold.y[idx]; prov=fold.prov_index[idx]; _,prov=np.unique(prov,return_inverse=True); nprov=prov.max()+1
    worst=0.0; checks=0
    for m in MODELS:
        p=np.load(os.path.join(OOF,f'{tag}_{m}_fold{fold.k}.npy')).astype('float64')[idx]
        yhat=(np.load(os.path.join(OOF,f'{tag}_{m}_fold{fold.k}_decision.npy'))[idx]>0).astype('int8') if m in ('svm','svc200') else (p>0.5).astype('int8')
        mf=ModelFold(p,yhat,y,prov); mf.prov_tp=prov[mf.tp]; mf.prov_fp=prov[mf.fp]; mf.prov_fn=prov[mf.fn]; mf.prov_pos_sorted=prov[mf.pos_sorted]
        for trial in range(3):
            cnt=np.ones(nprov) if trial==0 else rng.integers(0,4,size=nprov).astype('float64'); w=cnt[prov]
            f1,ap_,roc=mf.metrics(cnt); ref=(f1_score(y,yhat,sample_weight=w,zero_division=0),average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w))
            diffs=[abs(f1-ref[0]),abs(ap_-ref[1]),abs(roc-ref[2])]; assert not any(np.isnan(diffs)),(m,trial,diffs); worst=max(worst,max(diffs)); checks+=1
    return worst,checks
folds=[Fold(k) for k in range(1,6)]; print('loaded folds in %.0fs'%(time.time()-T0),flush=True)
worst,checks=validate(folds[0]); print(f'validation vs scikit-learn ({checks} checks, 200k rows, unit and random integer weights): max abs diff {worst:.2e}',flush=True); assert worst<1e-9
pe={m:{x:float(np.mean([f.models[m].metrics(np.ones(f.n_prov))[i] for f in folds])) for i,x in enumerate(METRICS)} for m in MODELS}
rp=os.path.join(RES,f'{tag}_reps.npz'); B=a.B
if os.path.exists(rp): z=np.load(rp); reps=z['reps']; done=int(z['done']); assert list(z['models'])==MODELS and reps.shape[0]==B
else: reps=np.full((B,len(MODELS),len(METRICS)),np.nan); done=0
rngs=[np.random.default_rng(SEED+f.k) for f in folds]
for b in range(done):
    for f,r in zip(folds,rngs): r.integers(0,f.n_prov,size=f.n_prov)          # replay the streams so a resumed run uses identical draws
t=time.time(); done0=done
for b in range(done,B):
    acc=np.zeros((len(MODELS),len(METRICS)))
    for f,r in zip(folds,rngs):
        cnt=np.bincount(r.integers(0,f.n_prov,size=f.n_prov),minlength=f.n_prov).astype('float64')
        for i,m in enumerate(MODELS): acc[i]+=f.models[m].metrics(cnt)
    reps[b]=acc/len(folds); done=b+1
    if done%50==0 or done==B:
        np.savez(rp,reps=reps,done=done,models=np.array(MODELS),metrics=np.array(METRICS)); print(f'{done}/{B} replicates ({time.time()-t:.0f}s)',flush=True)
        per50=(time.time()-t)/(done-done0)*50
        if done<B and (a.budget-(time.time()-T0))<per50*1.2+10: print('budget reached; resume later',flush=True); sys.exit(0)
assert not np.isnan(reps).any()
out={'tag':tag,'B':B,'seed':SEED,'definition':'provider-level bootstrap within folds (rng 2026+k), paired across models, mean over folds; percentile 2.5/97.5','validation_max_abs_diff_vs_sklearn':worst,'models':{},'differences':{}}
for i,m in enumerate(MODELS):
    out['models'][m]={x:{'point':pe[m][x],'boot_mean':float(np.mean(reps[:,i,j])),'ci_low':float(np.percentile(reps[:,i,j],2.5)),'ci_high':float(np.percentile(reps[:,i,j],97.5))} for j,x in enumerate(METRICS)}
for i,m1 in enumerate(MODELS):
    for i2,m2 in enumerate(MODELS):
        if i2<=i: continue
        out['differences'][f'{m1}-{m2}']={x:{'point':pe[m1][x]-pe[m2][x],'ci_low':float(np.percentile(reps[:,i,j]-reps[:,i2,j],2.5)),'ci_high':float(np.percentile(reps[:,i,j]-reps[:,i2,j],97.5)),'share_positive':float(np.mean(reps[:,i,j]>reps[:,i2,j]))} for j,x in enumerate(METRICS)}
json.dump(out,open(os.path.join(RES,f'{tag}.json'),'w'),indent=1)
L=[f'# Provider-level bootstrap 95% intervals ({tag}; B = {B}; seeds 2026+k); from saved OOF outputs only','| model | F1 (95% CI) | AUC-PR (95% CI) | AUC-ROC (95% CI) |','|---|---|---|---|']
for m in MODELS:
    v=out['models'][m]; L.append(f"| {m} | {v['f1']['point']:.4f} ({v['f1']['ci_low']:.4f}–{v['f1']['ci_high']:.4f}) | {v['auc_pr']['point']:.4f} ({v['auc_pr']['ci_low']:.4f}–{v['auc_pr']['ci_high']:.4f}) | {v['auc_roc']['point']:.3f} ({v['auc_roc']['ci_low']:.3f}–{v['auc_roc']['ci_high']:.3f}) |")
L+=['','| difference | ΔF1 (95% CI) | ΔAUC-PR (95% CI) | ΔAUC-ROC (95% CI) | share of replicates with ΔF1 > 0 |','|---|---|---|---|---|']
for d_,v in out['differences'].items(): L.append(f"| {d_} | {v['f1']['point']:+.4f} ({v['f1']['ci_low']:+.4f}–{v['f1']['ci_high']:+.4f}) | {v['auc_pr']['point']:+.4f} ({v['auc_pr']['ci_low']:+.4f}–{v['auc_pr']['ci_high']:+.4f}) | {v['auc_roc']['point']:+.3f} ({v['auc_roc']['ci_low']:+.3f}–{v['auc_roc']['ci_high']:+.3f}) | {v['f1']['share_positive']:.3f} |")
open(os.path.join(RES,f'{tag}.md'),'w').write('\n'.join(L)+'\n'); print('\n'.join(L))
