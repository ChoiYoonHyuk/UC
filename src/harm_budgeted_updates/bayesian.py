"""Equation-level Träuble belief updates, independent of evaluation responses."""
import numpy as np

def confusion(y,old,new,alpha=1.):
    y=np.asarray(y,int)
    assert y.ndim==1 and len(y)==len(old)==len(new) and np.all((y==0)|(y==1))
    matrices=[]
    for score in [old,new]:
        label=(np.asarray(score)>=.5).astype(int)
        counts=np.stack([np.bincount(label[y==c],minlength=2)for c in [0,1]],axis=1)
        matrices.append((counts+alpha)/(counts.sum(0,keepdims=True)+2*alpha))
    counts=np.bincount(y,minlength=2)
    prior=(counts+alpha)/(len(y)+2*alpha)
    return prior,np.stack(matrices)

def likelihood(score,matrix,soft):
    score=np.asarray(score,float)
    if soft:return (1-score[:,None])*matrix[0]+score[:,None]*matrix[1]
    return matrix[(score>=.5).astype(int)]

def beliefs(old,new,prior,matrices,soft):
    a=likelihood(old,matrices[0],soft)*prior
    initial=a/a.sum(1,keepdims=True)
    a=initial*likelihood(new,matrices[1],soft)
    final=a/a.sum(1,keepdims=True)
    assert np.all(np.isfinite(final)) and np.all(final>0)
    return initial[:,1],final[:,1]

def entropy(p):
    p=np.asarray(p,float)
    return -p*np.log(p)-(1-p)*np.log1p(-p)

def select(initial,ids,fraction,strategy,random_order):
    k=min(len(initial),max(1,int(np.ceil(fraction*len(initial)))))
    if strategy=='entropy':order=np.lexsort((ids,-entropy(initial)))
    elif strategy=='random':order=random_order
    else:raise ValueError(strategy)
    chosen=np.zeros(len(initial),bool);chosen[order[:k]]=True
    return chosen

def decisions(original_old,initial,final,chosen):
    old=(np.asarray(original_old)>=.5).astype(np.uint8)
    candidate=(np.asarray(final)>.5).astype(np.uint8) # binary argmax returns zero at an exact tie
    out={'MB':np.where(chosen,candidate,old)}
    out['MBME']=np.where(chosen&(entropy(final)<entropy(initial)),candidate,old)
    p_candidate=np.where(candidate==1,final,1-final)
    p_old=np.where(old==1,final,1-final)
    for ratio in [1.,5.,10.]:
        change=chosen&(p_candidate>ratio*p_old)
        out['CR'+str(int(ratio))]=np.where(change,candidate,old)
    return out
