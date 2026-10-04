#!/usr/bin/env python3
"""Gate 3 summaries for the CMS-LEIE arm, computed only from saved outputs (results/<set>_<label>/<model>/fold<k>.json and
results/oof/*.npy): per-model mean ± SD over folds; metrics recomputed from the stored probabilities (consistency check);
the soft-voting ensemble from its members' stored probabilities; rank orders by F1 and AUC-PR with Kendall tau against the
paper's Tables 1-2 (saved experiment_results.json of the synthetic and Kaggle arms); Gradient Boosting threshold study from
the pooled out-of-fold probabilities (natural prevalence) with the precision identity check and nested selection.
Usage: summarise.py [--set primary] [--label is_fraud]"""
import os, sys, json, glob, argparse, numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, average_precision_score, roc_auc_score
from scipy.stats import kendalltau, spearmanr
ap=argparse.ArgumentParser(); ap.add_argument('--set',default='primary'); ap.add_argument('--label',default='is_fraud'); ap.add_argument('--ratio',type=int,default=20); a=ap.parse_args()
HERE=os.path.dirname(os.path.abspath(__file__)); tag=f'{a.set}_{a.label}'+('' if a.ratio==20 else f'_r{a.ratio}'); RES=os.path.join(HERE,'results',tag); OOF=os.path.join(HERE,'results','oof'); DATA=os.path.join(HERE,'data')
NAMES={'rf':'Random Forest','svm':'SVM (RBF)','mlp':'Neural Network (MLP)','gb':'Gradient Boosting','nb':'Naive Bayes','ens':'AdaBoost-Voting Ensemble'}
UP="/mnt/user-data/uploads/EB1A prep/GitHub_Framework/results"
def paper(arm):
    p=os.path.join(UP,arm,'experiment_results.json')
    if not os.path.exists(p): return None
    d=json.load(open(p)); r=d.get('results',d); return {m:{'f1':v['f1']['mean'],'auc_pr':v['auc_pr']['mean']} for m,v in r.items() if isinstance(v,dict) and 'f1' in v}
out={'set':a.set,'label':a.label,'models':{},'folds_available':{}}
ys={}; preds={}
for k in range(1,6):
    yp=os.path.join(DATA,f'test_{a.set}_{k}_y_{a.label}.npy')
    if os.path.exists(yp): ys[k]=np.load(yp)
def load_pred(m,k):
    p=os.path.join(OOF,f'{tag}_{m}_fold{k}.npy'); return np.load(p) if os.path.exists(p) else None
def metrics(y,p,yhat):
    return {'f1':float(f1_score(y,yhat)),'precision':float(precision_score(y,yhat,zero_division=0)),'recall':float(recall_score(y,yhat)),'mcc':float(matthews_corrcoef(y,yhat)),
            'auc_pr':float(average_precision_score(y,p)),'auc_roc':float(roc_auc_score(y,p))}
for m in ['gb','rf','mlp','nb','svm']:
    folds={}
    for k in range(1,6):
        jp=os.path.join(RES,m,f'fold{k}.json')
        if not os.path.exists(jp): continue
        j=json.load(open(jp)); p=load_pred(m,k); y=ys[k]
        if m=='svm':
            dec=np.load(os.path.join(OOF,f'{tag}_svm_fold{k}_decision.npy')); yhat=(dec>0).astype(int)
        else: yhat=(p>0.5).astype(int)
        chk=metrics(y,p,yhat); diff=max(abs(chk[x]-j[x]) for x in chk)
        folds[k]={**{x:j[x] for x in ['f1','precision','recall','mcc','auc_pr','auc_roc','tp','fp','fn','tn','fit_seconds']},'recomputed_max_abs_diff':float(diff)}
    if folds:
        out['models'][m]={'name':NAMES[m],'folds':folds,'n_folds':len(folds)}
        for x in ['f1','precision','recall','mcc','auc_pr','auc_roc']:
            v=[folds[k][x] for k in folds]; out['models'][m][x]={'mean':float(np.mean(v)),'std':float(np.std(v))}
# ensemble from members (rf200, gb200, svc200)
ens={}
for k in range(1,6):
    ps=[load_pred(mm,k) for mm in ['rf200','gb200','svc200']]
    if all(p is not None for p in ps) and k in ys:
        p=np.mean(ps,axis=0); y=ys[k]; yhat=(p>0.5).astype(int); ens[k]=metrics(y,p,yhat)
        np.save(os.path.join(OOF,f'{tag}_ens_fold{k}.npy'),p.astype('float32'))
if ens:
    out['models']['ens']={'name':NAMES['ens'],'folds':ens,'n_folds':len(ens)}
    for x in ['f1','precision','recall','mcc','auc_pr','auc_roc']:
        v=[ens[k][x] for k in ens]; out['models']['ens'][x]={'mean':float(np.mean(v)),'std':float(np.std(v))}
# rank orders and comparison with the paper's arms (models with all 5 folds only)
complete=[m for m in out['models'] if out['models'][m]['n_folds']==5]
out['complete_models']=complete
for metric in ['f1','auc_pr']:
    order=sorted(complete,key=lambda m:-out['models'][m][metric]['mean']); out[f'rank_{metric}']=[NAMES[m] for m in order]
    comp={}
    for arm in ['synthetic','kaggle_medicare']:
        pp=paper(arm)
        if pp and len(complete)>=3:
            names=[NAMES[m] for m in complete]; ours=[out['models'][m][metric]['mean'] for m in complete]; theirs=[pp[n][metric] for n in names]
            tau=kendalltau(ours,theirs); rho=spearmanr(ours,theirs)
            comp[arm]={'models':names,'cms_leie':ours,'paper':theirs,'kendall_tau':float(tau.statistic),'kendall_p':float(tau.pvalue),'spearman_rho':float(rho.statistic),'paper_rank':[n for _,n in sorted(zip(theirs,names),reverse=True)]}
    out[f'rank_comparison_{metric}']=comp
# GB threshold study from pooled OOF (natural prevalence)
if 'gb' in out['models'] and out['models']['gb']['n_folds']==5:
    y=np.concatenate([ys[k] for k in range(1,6)]); p=np.concatenate([load_pred('gb',k) for k in range(1,6)]); pi=float(y.mean())
    def at(t,yy,pp):
        yh=(pp>t).astype(int); tp=int(((yh==1)&(yy==1)).sum()); fp=int(((yh==1)&(yy==0)).sum()); fn=int(((yh==0)&(yy==1)).sum()); tn=int(((yh==0)&(yy==0)).sum())
        P=tp/(tp+fp) if tp+fp else 0.0; R=tp/(tp+fn) if tp+fn else 0.0; F=2*P*R/(P+R) if P+R else 0.0; FPR=fp/(fp+tn) if fp+tn else 0.0
        ident=(pi*R)/(pi*R+(1-pi)*FPR) if (pi*R+(1-pi)*FPR)>0 else 0.0
        return {'threshold':t,'precision':P,'recall':R,'f1':F,'fpr':FPR,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'flagged_share':(tp+fp)/len(yy),'precision_identity':ident,'identity_abs_diff':abs(P-ident)}
    rows=[at(t,y,p) for t in [0.3,0.5,0.6,0.7,0.8,0.9,0.95]]
    grid=np.round(np.arange(0.01,1.0,0.01),2); gf=[(t,at(t,y,p)['f1']) for t in grid]; tbest=max(gf,key=lambda z:z[1])
    nested=[]
    for k in range(1,6):
        ytr=np.concatenate([ys[j] for j in range(1,6) if j!=k]); ptr=np.concatenate([load_pred('gb',j) for j in range(1,6) if j!=k])
        tk=max(((t,at(t,ytr,ptr)['f1']) for t in grid),key=lambda z:z[1])[0]; r=at(tk,ys[k],load_pred('gb',k)); r['chosen_threshold']=tk; nested.append(r)
    out['gb_threshold_study']={'prevalence':pi,'rows':rows,'grid_optimum':{'threshold':float(tbest[0]),'f1':float(tbest[1]),**at(float(tbest[0]),y,p)},
        'nested':{'per_fold':nested,'f1_mean':float(np.mean([r['f1'] for r in nested])),'f1_std':float(np.std([r['f1'] for r in nested])),'thresholds':[r['chosen_threshold'] for r in nested]},
        'max_identity_abs_diff':max(r['identity_abs_diff'] for r in rows)}
json.dump(out,open(os.path.join(RES,'summary.json'),'w'),indent=1)
# markdown
L=[f'# CMS–LEIE arm — summary ({tag}); generated from saved fold outputs\n','| model | folds | F1 | precision | recall | MCC | AUC-PR | AUC-ROC |','|---|---|---|---|---|---|---|---|']
for m,v in out['models'].items():
    L.append(f"| {v['name']} | {v['n_folds']} | {v['f1']['mean']:.4f} ± {v['f1']['std']:.4f} | {v['precision']['mean']:.4f} | {v['recall']['mean']:.4f} | {v['mcc']['mean']:.4f} | {v['auc_pr']['mean']:.4f} ± {v['auc_pr']['std']:.4f} | {v['auc_roc']['mean']:.4f} ± {v['auc_roc']['std']:.4f} |")
for metric in ['f1','auc_pr']:
    if f'rank_{metric}' in out:
        L.append(f"\nRank by {metric} (complete models): "+' > '.join(out[f'rank_{metric}']))
        for arm,c in out.get(f'rank_comparison_{metric}',{}).items(): L.append(f"- vs paper {arm}: Kendall τ = {c['kendall_tau']:.2f} (p = {c['kendall_p']:.2f}), Spearman ρ = {c['spearman_rho']:.2f}; paper order: "+' > '.join(c['paper_rank']))
if 'gb_threshold_study' in out:
    g=out['gb_threshold_study']; L.append(f"\n## Gradient Boosting threshold study (pooled OOF, prevalence {g['prevalence']*100:.4f}%)\n| threshold | precision | recall | F1 | FPR | flagged share | identity |Δ| |\n|---|---|---|---|---|---|---|")
    for r in g['rows']: L.append(f"| {r['threshold']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['fpr']*100:.3f}% | {r['flagged_share']*100:.3f}% | {r['identity_abs_diff']:.2e} |")
    L.append(f"Grid optimum {g['grid_optimum']['threshold']:.2f} (F1 {g['grid_optimum']['f1']:.4f}, P {g['grid_optimum']['precision']:.4f}, R {g['grid_optimum']['recall']:.4f}, FPR {g['grid_optimum']['fpr']*100:.3f}%); nested {g['nested']['f1_mean']:.4f} ± {g['nested']['f1_std']:.4f} (thresholds {g['nested']['thresholds']}).")
open(os.path.join(RES,'summary.md'),'w').write('\n'.join(L)+'\n'); print('\n'.join(L))
