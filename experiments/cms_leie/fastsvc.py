"""Exact, fast replacement for sklearn SVC (RBF, binary, probability=True) decision_function and predict_proba.
decision: K(X, SV) @ dual_coef + intercept with the RBF kernel computed by BLAS (matches libsvm to ~1e-9).
proba: libsvm's svm_predict_probability as bundled with scikit-learn — Platt sigmoid of the (sign-flipped) decision value,
clipped to [1e-7, 1-1e-7], then the Wu–Lin–Weng multiclass_probability iteration (eps = 0.005/k, max_iter = max(100,k)),
which scikit-learn's libsvm runs even for two classes (this is why predict_proba is not exactly the sigmoid of
decision_function). Ported line by line from svm.cpp and vectorised over samples with per-sample convergence masks, so the
per-sample iteration sequence is identical."""
import numpy as np
def decision_function(est, X, chunk=1000):
    SV=est.support_vectors_; dc=est.dual_coef_[0]; b=est.intercept_[0]; g=est._gamma
    sv2=(SV**2).sum(1); out=np.empty(len(X))
    for s in range(0,len(X),chunk):
        x=np.asarray(X[s:s+chunk],dtype='float64'); d2=(x**2).sum(1)[:,None]+sv2[None,:]-2*x@SV.T; np.maximum(d2,0,out=d2)
        out[s:s+chunk]=np.exp(-g*d2)@dc+b
    return out
def _sigmoid_predict(dec,A,B):
    fApB=dec*A+B; out=np.empty_like(fApB); pos=fApB>=0
    out[pos]=np.exp(-fApB[pos])/(1.0+np.exp(-fApB[pos])); out[~pos]=1.0/(1+np.exp(fApB[~pos])); return out
def _multiclass_probability_k2(r01):
    """libsvm multiclass_probability for k=2, vectorised. r01 = pairwise_prob[0][1]; r10 = 1 - r01. Returns p[:,0], p[:,1]."""
    n=len(r01); k=2; eps=0.005/k; max_iter=max(100,k)
    r=np.zeros((n,2,2)); r[:,0,1]=r01; r[:,1,0]=1.0-r01
    Q=np.zeros((n,2,2))
    for t in range(k):
        for j in range(t):
            Q[:,t,t]+=r[:,j,t]*r[:,j,t]; Q[:,t,j]=Q[:,j,t]
        for j in range(t+1,k):
            Q[:,t,t]+=r[:,j,t]*r[:,j,t]; Q[:,t,j]=-r[:,j,t]*r[:,t,j]
    p=np.full((n,2),1.0/k); active=np.ones(n,dtype=bool)
    for it in range(max_iter):
        Qp=np.einsum('ntj,nj->nt',Q,p); pQp=(p*Qp).sum(1)
        max_error=np.abs(Qp-pQp[:,None]).max(1)
        active&=~(max_error<eps)
        if not active.any(): break
        a=active
        for t in range(k):
            diff=(-Qp[a,t]+pQp[a])/Q[a,t,t]
            p[a,t]+=diff
            pQp[a]=(pQp[a]+diff*(diff*Q[a,t,t]+2*Qp[a,t]))/(1+diff)/(1+diff)
            for j in range(k):
                Qp[a,j]=(Qp[a,j]+diff*Q[a,t,j])/(1+diff); p[a,j]/=(1+diff)
    return p[:,0],p[:,1]
def predict_proba1(est, dec):
    """Probability of class 1 from the sklearn decision_function values (dec = -libsvm dec_value for binary C-SVC)."""
    A=est.probA_[0]; B=est.probB_[0]; min_prob=1e-7
    dec_libsvm=-dec
    pair=np.minimum(np.maximum(_sigmoid_predict(dec_libsvm,A,B),min_prob),1-min_prob)   # pairwise_prob[0][1]
    p0,p1=_multiclass_probability_k2(pair)
    return p1
