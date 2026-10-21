"""Mathematically equivalent binary response metrics with one shared score order."""
import numpy as np
from ._binary import dinfinity_binary
class BinaryMetricContext:
 def __init__(self,b,y):
  self.b=np.asarray(b,np.float64);self.y=np.asarray(y,int);assert self.b.shape==self.y.shape and self.y.ndim==1
  self.n=len(y);self.positive=int(self.y.sum());assert 0<self.positive<self.n
  self.k=max(1,int(np.ceil(.01*self.n)));self.old_loss=-np.where(self.y==1,np.log(self.b),np.log1p(-self.b));self.old_nll=float(self.old_loss.mean());self.before=np.zeros(self.n,bool);self.before[np.argsort(-self.b,kind='stable')[:self.k]]=True;self.oldtp=int(np.sum(self.before&(self.y==1)));self.bc={t:(self.b>=t)==self.y for t in [.1,.5]};self.class_masks={c:self.y==c for c in [0,1]}
 def __call__(self,p):
  p=np.asarray(p,np.float64);b,y,n,k=self.b,self.y,self.n,self.k;assert p.shape==b.shape
  loss=-np.where(y==1,np.log(p),np.log1p(-p));d=loss-self.old_loss;harm=np.maximum(d,0);cost=dinfinity_binary(b,p);order=np.argsort(-p,kind='stable');s=p[order];yy=y[order];ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),n-1];tp=np.cumsum(yy)[ends];pos=np.diff(np.r_[0,tp]);cnt=np.diff(np.r_[0,ends+1]);neg=cnt-pos;before_neg=np.r_[0,np.cumsum(neg)[:-1]];AP=float(np.sum(pos/self.positive*(tp/(ends+1))));AUROC=float(np.sum(pos*((n-self.positive)-before_neg-neg/2))/(self.positive*(n-self.positive)))
  after=np.zeros(n,bool);after[order[:k]]=True;enter=int(np.sum(after&~self.before&(y==1)));lost=int(np.sum(self.before&~after&(y==1)));newtp=int(np.sum(after&(y==1)));confidence=np.maximum(p,1-p);correct=((p>=.5)==y).astype(float);bins=np.minimum((15*confidence).astype(int),14);ece=float(np.sum(abs(np.bincount(bins,weights=correct,minlength=15)-np.bincount(bins,weights=confidence,minlength=15)))/n)
  out=dict(n_test=n,positives=self.positive,prevalence=float(y.mean()),reference_nll=self.old_nll,nll=float(loss.mean()),gain=float(self.old_nll-loss.mean()),H=float(harm.mean()),B=float(np.maximum(-d,0).mean()),C=float(cost.mean()),max_cost=float(cost.max()),tail1=float(np.partition(harm,n-k)[-k:].mean()),max_harm=float(harm.max()),AP=AP,AUROC=AUROC,Brier=float(np.mean((p-y)**2)),ECE15=ece,queue_capacity=k,queue_enter_TP=enter,queue_lost_TP=lost,queue_net_TP=newtp-self.oldtp,queue_enter_over_capacity=enter/k,queue_lost_over_capacity=lost/k,queue_net_over_capacity=(newtp-self.oldtp)/k,precision_at_capacity=newtp/k,recall_at_capacity=newtp/self.positive,changed_fraction=float(np.mean(abs(p-b)>1e-12)))
  for c,mask in self.class_masks.items():out['H'+str(c)]=float(harm[mask].mean());out['nll'+str(c)]=float(loss[mask].mean())
  for threshold in [.1,.5]:
   bc=self.bc[threshold];pc=(p>=threshold)==y;neg=bc&~pc;pos=~bc&pc;suffix=str(threshold);out['negative_flip_'+suffix]=float(neg.mean());out['positive_flip_'+suffix]=float(pos.mean())
   for c,mask in self.class_masks.items():out['negative_flip_class'+str(c)+'_'+suffix]=float(neg[mask].mean());out['positive_flip_class'+str(c)+'_'+suffix]=float(pos[mask].mean())
   if threshold==.1:
    fp=int(np.sum((p>=threshold)&(y==0)));fn=int(np.sum((p<threshold)&(y==1)));out.update(action_FP=fp,action_FN=fn,action_cost=(fp+9*fn)/n)
  return out
