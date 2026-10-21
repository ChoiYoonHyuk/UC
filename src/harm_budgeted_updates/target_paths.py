"""Probability-only target paths and exact sorted-capacity UC water filling."""
import time
import numpy as np
from scipy.special import expit,logit
from ._binary import dinfinity_binary,floor_bounds,_at_multiplier

def shrink(b,t,path,weight):
    if weight==1.:return t.copy()
    if weight==0.:return b.copy()
    if path=='Probability':return b+weight*(t-b)
    if path=='Logit':return expit(logit(b)+weight*(logit(t)-logit(b)))
    raise ValueError(path)

def uniform_sorted(b,t,h,tau=.1):
    start=time.perf_counter();end=np.clip(t,*floor_bounds(b,tau));capacities=dinfinity_binary(b,end)
    if h==0. or not np.any(capacities):
        q=b.copy();gamma=0.;location='identity';repairs=0
    elif capacities.mean()<=h:
        q=end;gamma=float(capacities.max());location='endpoint';repairs=0
    else:
        a=np.sort(capacities);s=np.cumsum(a);n=len(a)
        levels=(s+(n-np.arange(1,n+1))*a)/n
        j=int(np.searchsorted(levels,h,side='left'));assert j<n
        previous=0. if j==0 else s[j-1]
        gamma=float((n*h-previous)/(n-j));gamma=min(tau,max(0.,gamma))
        q=np.clip(t,*floor_bounds(b,gamma));repairs=0
        # Stable floor calculations can move an analytical water level by ulps.
        while dinfinity_binary(b,q).mean()>h:
            gamma=max(0.,gamma-max(5e-16,1e-13*gamma));q=np.clip(t,*floor_bounds(b,gamma));repairs+=1
            if repairs>16:raise ArithmeticError('Sorted UC numerical feasibility repair failed')
        location='interior'
    cost=dinfinity_binary(b,q)
    assert cost.mean()<=h+2e-16 and cost.max()<=tau+2e-14
    return q,dict(method='uniform_cap_sorted_capacity',parameter=gamma,canonical_gamma=gamma,
                  location=location,allowance=h,cap=tau,mean_cost=float(cost.mean()),maximum_cost=float(cost.max()),
                  endpoint_mean_cost=float(capacities.mean()),endpoint_maximum_cost=float(capacities.max()),
                  numerical_repairs=repairs,seconds=time.perf_counter()-start)

def restore(b,t,method,record,tau=.1):
    if method=='Input':return b.copy()
    if method=='Native':return t.copy()
    if method=='CapOnly':return np.clip(t,*floor_bounds(b,tau))
    if method=='UC':
        if record['location']=='identity':return b.copy()
        if record['location']=='endpoint':return np.clip(t,*floor_bounds(b,tau))
        return np.clip(t,*floor_bounds(b,record['parameter']))
    lower,upper=floor_bounds(b,record['effective_gamma'])
    if record['location']=='identity':return b.copy()
    if record['location']=='endpoint':return np.clip(t,lower,upper)
    return _at_multiplier(b,t,record['multiplier'],lower,upper)
