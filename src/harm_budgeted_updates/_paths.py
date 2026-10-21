"""Two label-free scalar control paths, with original per-case cap."""
import numpy as np
from ._binary import floor_bounds,dinfinity_binary
def matched_path(reference,target,allowance,tau=.1,kind='uniform_cap'):
 b=np.asarray(reference,float);t=np.asarray(target,float);h=float(allowance)
 if b.ndim!=1 or b.shape!=t.shape or not np.isfinite(h)or h<0 or not np.isfinite(b).all()or not np.isfinite(t).all():raise ValueError('Finite matching binary probabilities and nonnegative cost required')
 if kind not in ['uniform_cap','native_forward_KL']:raise ValueError('Prespecified path required')
 if np.any((b<=0)|(b>=1)|(t<0)|(t>1)):raise ValueError('Open reference and probability target required')
 lower,upper=floor_bounds(b,tau);end=np.clip(t,lower,upper)
 def at(value):
  if kind=='uniform_cap':
   lo,hi=floor_bounds(b,value);return np.clip(t,lo,hi)
  if kind=='native_forward_KL':return np.clip(b+value*(t-b),lower,upper)
  raise ValueError('Prespecified path required')
 maxparam=float(tau)if kind=='uniform_cap'else 1.
 if h==0:p=b.copy();param=0.
 elif float(dinfinity_binary(b,end).mean())<=h:p=end;param=maxparam
 else:
  low,high=0.,maxparam
  for _ in range(80):
   middle=low+(high-low)/2
   if middle==low or middle==high:break
   if float(dinfinity_binary(b,at(middle)).mean())<=h:low=middle
   else:high=middle
  param=low;p=at(param)
 return p,dict(method=kind,parameter=float(param),mean_cost=float(dinfinity_binary(b,p).mean()),allowance=h,maximum_cost=float(dinfinity_binary(b,p).max()),cap=tau)
def uniform_cap_binary(reference,target,allowance,tau=.1):return matched_path(reference,target,allowance,tau,'uniform_cap')
def native_forward_kl_binary(reference,target,allowance,tau=.1):return matched_path(reference,target,allowance,tau,'native_forward_KL')
