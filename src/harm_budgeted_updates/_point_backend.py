"""Double-precision Point allocation; GPU dual search with CPU final reconstruction."""
import time
import numpy as np
from ._binary import allocate_binary,floor_bounds,dinfinity_binary,_at_multiplier,_record

def allocate_fast(reference,target,allowance,tau=.1,force_gpu=False):
 b=np.asarray(reference,np.float64);t=np.asarray(target,np.float64);h=float(allowance)
 if len(b)<100000 and not force_gpu:return allocate_binary(b,t,h,tau=tau)
 import torch
 if not torch.cuda.is_available():return allocate_binary(b,t,h,tau=tau)
 start=time.perf_counter();lower,upper=floor_bounds(b,tau);end=np.clip(t,lower,upper);w=np.full(len(b),1/len(b));ec=float(w@dinfinity_binary(b,end))
 if h==0 or ec==0:
  q=b.copy();loc='identity';value=None;lo=hi=None;iterations=0
 elif ec<=h:
  q=end;loc='endpoint';value=0.;lo=hi=0.;iterations=0
 else:
  tb=torch.as_tensor(b,device='cuda',dtype=torch.float64);tt=torch.as_tensor(t,device='cuda',dtype=torch.float64);tl=torch.as_tensor(lower,device='cuda',dtype=torch.float64);tu=torch.as_tensor(upper,device='cuda',dtype=torch.float64)
  def at(lam):
   inv=1/(1+lam);return torch.minimum(torch.maximum(torch.minimum(torch.maximum(tb,tt*inv),(tt+lam)*inv),tl),tu)
  def mean_cost(p):
   change=p-tb
   up=-torch.log1p(-change/(1-tb));down=-torch.log1p(change/tb)
   return float(torch.where(change>0,up,torch.where(change<0,down,torch.zeros_like(p))).mean().item())
  hsolve=max(0.,h-max(2e-14,1e-11*h))
  lo,hi=0.,1.
  while mean_cost(at(hi))>hsolve:lo=hi;hi*=2
  iterations=0
  for _ in range(100):
   mid=lo+(hi-lo)/2
   if mid==lo or mid==hi:break
   iterations+=1
   if mean_cost(at(mid))<=hsolve:hi=mid
   else:lo=mid
  value=hi;q=_at_multiplier(b,t,value,lower,upper);loc='interior';del tb,tt,tl,tu
  # Reduction rounding can place a GPU bracket within ulps of the CPU boundary.
  for _ in range(64):
   if float(w@dinfinity_binary(b,q))<=h:break
   value=float(np.nextafter(value+max(1.,abs(value))*1e-10,np.inf));q=_at_multiplier(b,t,value,lower,upper)
  else:raise ArithmeticError('CPU feasibility repair failed')
 assert float(w@dinfinity_binary(b,q))<=h
 rec=_record(b,t,q,w,h,lower,upper,method='allocation',multiplier=value,location=loc,dual_lower=lo,dual_upper=hi,iterations=iterations,backend='torch_cuda_float64_dual_cpu_output',target_duality_gap_upper=float((value or 0.)*max(0.,h-float(w@dinfinity_binary(b,q)))),numerical_allowance_margin=max(2e-14,1e-11*h),seconds=time.perf_counter()-start)
 return q,rec
