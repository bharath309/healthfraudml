#!/usr/bin/env python3
"""Resumable, time-budgeted training + test-fold prediction for one (model, fold) of the CMS-LEIE arm (protocol_frozen.md).
Usage: train_model.py MODEL FOLD [--set primary] [--label is_fraud] [--budget 450] [--ratio 20]
MODEL in: gb, rf, nb, mlp, svm, rf200, gb200, svc200   (rf200/gb200/svc200 are the soft-voting ensemble's members)
State is pickled after every increment in state/; on completion the test fold is predicted in batches (also resumable) and
results/oof/<set>_<label>_<MODEL>_fold<K>.npy (probability of class 1; for svm/svc200 a second array with the decision
function, because SVC.predict uses the decision function, not the Platt probabilities) and results/<set>_<label>/<MODEL>/
fold<K>.json are written. Hyper-parameters are those of run_comparison.run_experiments (random_state=42)."""
import os, sys, time, json, pickle, argparse, numpy as np, pandas as pd, cmsarm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
RS=42
ap=argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('fold',type=int); ap.add_argument('--set',default='primary'); ap.add_argument('--label',default='is_fraud'); ap.add_argument('--budget',type=float,default=450); ap.add_argument('--ratio',type=int,default=20)
a=ap.parse_args(); T0=time.time(); k=a.fold; tag=f'{a.set}_{a.label}'+('' if a.ratio==20 else f'_r{a.ratio}')
STATE=os.path.join(cmsarm.HERE,'state'); RES=os.path.join(cmsarm.HERE,'results',tag,a.model); OOF=os.path.join(cmsarm.HERE,'results','oof'); os.makedirs(STATE,exist_ok=True); os.makedirs(RES,exist_ok=True); os.makedirs(OOF,exist_ok=True)
sp=os.path.join(STATE,f'{tag}_{a.model}_fold{k}.pkl'); done_json=os.path.join(RES,f'fold{k}.json')
if os.path.exists(done_json): print('done already:',done_json); sys.exit(0)
def left(): return a.budget-(time.time()-T0)
st=pickle.load(open(sp,'rb')) if os.path.exists(sp) else {'phase':'train','fit_seconds':0.0,'increments':[]}
def save(): pickle.dump(st,open(sp,'wb'))
# ---------------- training ----------------
if st['phase']=='train':
    d=np.load(os.path.join(cmsarm.DATA,f'resampled_{a.set}_{a.label}_{k}'+('' if a.ratio==20 else f'_r{a.ratio}')+'.npz')); X=d['X'].astype('float64'); y=d['y'].astype('int64')
    st.setdefault('train_rows',int(len(y))); st.setdefault('train_pos',int(y.sum()))
    if a.model in ('gb','rf','rf200','gb200'):
        total={'gb':500,'rf':500,'rf200':200,'gb200':200}[a.model]
        if 'est' not in st:
            if a.model=='gb': st['est']=GradientBoostingClassifier(n_estimators=0,max_depth=6,learning_rate=0.05,subsample=0.8,random_state=RS,warm_start=True)
            elif a.model=='gb200': st['est']=GradientBoostingClassifier(n_estimators=0,max_depth=5,learning_rate=0.05,random_state=RS,warm_start=True)
            elif a.model=='rf': st['est']=RandomForestClassifier(n_estimators=0,max_depth=20,min_samples_leaf=5,class_weight='balanced',random_state=RS,n_jobs=-1,warm_start=True)
            else: st['est']=RandomForestClassifier(n_estimators=0,max_depth=15,class_weight='balanced',random_state=RS,n_jobs=-1,warm_start=True)
            st['n']=0
        step={'gb':30,'gb200':30,'rf':100,'rf200':100}[a.model]; est=st['est']
        while st['n']<total:
            inc=min(step,total-st['n']); est.set_params(n_estimators=st['n']+inc); t=time.time(); est.fit(X,y); dt=time.time()-t
            st['n']+=inc; st['fit_seconds']+=dt; st['increments'].append(round(dt,1)); save()
            print(f'{a.model} fold{k}: {st["n"]}/{total} estimators (+{inc} in {dt:.0f}s), fit total {st["fit_seconds"]:.0f}s',flush=True)
            if st['n']<total and left()<dt*1.15+20: print('budget reached; resume later',flush=True); sys.exit(0)
        st['phase']='predict'; save()
    else:
        if a.model in ('svm','svc200'):
            rng=np.random.default_rng(1000+k); idx=rng.choice(len(y),size=min(30000,len(y)),replace=False); Xs,ys=X[idx],y[idx]; st['svc_subsample_rows']=int(len(idx)); st['svc_subsample_pos']=int(ys.sum())
            est=SVC(kernel='rbf',C=10,gamma='scale',class_weight='balanced',probability=True,random_state=RS)
        elif a.model=='mlp': est=MLPClassifier(hidden_layer_sizes=(128,64,32),activation='relu',max_iter=200,early_stopping=True,validation_fraction=0.1,random_state=RS); Xs,ys=X,y
        elif a.model=='nb': est=GaussianNB(); Xs,ys=X,y
        else: raise SystemExit('unknown model '+a.model)
        t=time.time(); est.fit(Xs,ys); st['fit_seconds']=time.time()-t; st['est']=est; st['phase']='predict'
        if a.model=='mlp': st['mlp_n_iter']=int(est.n_iter_); st['mlp_best_val']=float(est.best_validation_score_) if hasattr(est,'best_validation_score_') else None
        if a.model in ('svm','svc200'): st['n_support']=[int(v) for v in est.n_support_]
        save(); print(f'{a.model} fold{k}: fitted in {st["fit_seconds"]:.0f}s',flush=True)
# ---------------- prediction on the complete test fold (batched, resumable) ----------------
if st['phase']=='predict':
    est=st['est']; keyp=os.path.join(OOF,f'keys_fold{k}.parquet'); Xp=os.path.join(cmsarm.DATA,f'test_{a.set}_{k}.npy'); yp=os.path.join(cmsarm.DATA,f'test_{a.set}_{k}_y_{a.label}.npy')
    if not (os.path.exists(Xp) and os.path.exists(yp)):
        Xt,yt,nt,yrt=cmsarm.load_test(k,a.set,a.label); np.save(Xp,Xt.astype('float32')); np.save(yp,yt.astype('int8'))
        if not os.path.exists(keyp): pd.DataFrame({'npi':nt,'year':yrt}).to_parquet(keyp,index=False)
        del Xt
    Xt=np.load(Xp,mmap_mode='r'); n=Xt.shape[0]; B=250000
    pp=os.path.join(STATE,f'{tag}_{a.model}_fold{k}_pred.npy'); dp=os.path.join(STATE,f'{tag}_{a.model}_fold{k}_dec.npy')
    pred=np.load(pp) if os.path.exists(pp) else np.full(n,np.nan,dtype='float32'); dec=(np.load(dp) if os.path.exists(dp) else np.full(n,np.nan,dtype='float32')) if a.model in ('svm','svc200') else None
    st.setdefault('pred_seconds',0.0)
    for s in range(0,n,B):
        if not np.isnan(pred[s]): continue
        t=time.time(); xb=np.asarray(Xt[s:s+B],dtype='float64')
        if dec is not None:   # SVC: exact BLAS kernel path (fastsvc.py; validated 3 Oct: decision 2.6e-9, proba 3.3e-10 vs libsvm)
            import fastsvc; f=fastsvc.decision_function(est,xb); dec[s:s+B]=f.astype('float32'); pred[s:s+B]=fastsvc.predict_proba1(est,f).astype('float32')
        else:
            pred[s:s+B]=est.predict_proba(xb)[:,1].astype('float32')
        st['pred_seconds']+=time.time()-t; np.save(pp,pred);
        if dec is not None: np.save(dp,dec)
        save(); print(f'{a.model} fold{k}: predicted {min(s+B,n)}/{n} ({time.time()-t:.0f}s)',flush=True)
        if s+B<n and left()<(time.time()-t)*1.3+15: print('budget reached in prediction; resume later',flush=True); sys.exit(0)
    assert not np.isnan(pred).any()
    np.save(os.path.join(OOF,f'{tag}_{a.model}_fold{k}.npy'),pred)
    if dec is not None: np.save(os.path.join(OOF,f'{tag}_{a.model}_fold{k}_decision.npy'),dec)
    yt=np.load(yp)
    from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, average_precision_score, roc_auc_score, confusion_matrix
    if a.model in ('svm','svc200'): yhat=(dec>0).astype(int)          # SVC.predict uses the decision function
    else: yhat=(pred>0.5).astype(int)                                # GB: score>0 <=> p>0.5; RF/NB/MLP: argmax, ties -> class 0
    tn,fp,fn,tp=confusion_matrix(yt,yhat,labels=[0,1]).ravel()
    import sklearn, imblearn
    rec={'model':a.model,'fold':k,'set':a.set,'label':a.label,'train_rows_resampled':st.get('train_rows'),'train_pos_resampled':st.get('train_pos'),'test_rows':int(len(yt)),'test_pos':int(yt.sum()),
         'f1':float(f1_score(yt,yhat)),'precision':float(precision_score(yt,yhat,zero_division=0)),'recall':float(recall_score(yt,yhat)),'mcc':float(matthews_corrcoef(yt,yhat)),
         'auc_pr':float(average_precision_score(yt,pred)),'auc_roc':float(roc_auc_score(yt,pred)),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
         'fit_seconds':round(st['fit_seconds'],1),'pred_seconds':round(st['pred_seconds'],1),'increments':st.get('increments'),'svc_subsample_rows':st.get('svc_subsample_rows'),'svc_subsample_pos':st.get('svc_subsample_pos'),'n_support':st.get('n_support'),'mlp_n_iter':st.get('mlp_n_iter'),
         'versions':{'sklearn':sklearn.__version__,'imblearn':imblearn.__version__,'numpy':np.__version__}}
    json.dump(rec,open(done_json,'w'),indent=1); print(json.dumps({x:rec[x] for x in ['model','fold','f1','precision','recall','mcc','auc_pr','auc_roc','tp','fp','fn','fit_seconds']}),flush=True)
    for p in (pp,dp):
        if p and os.path.exists(p): os.remove(p)
    os.remove(sp)
