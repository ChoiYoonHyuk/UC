"""Finite binomial confidence intervals, computed without response probabilities."""
import math
import numpy as np

def binomial_cdf(x,n,p):
    if x<0:return 0.
    if x>=n:return 1.
    if p<=0:return 1.
    if p>=1:return 0.
    k=np.arange(x+1,dtype=float)
    logchoose=np.array([math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)for j in range(x+1)])
    logs=logchoose+k*np.log(p)+(n-k)*np.log1p(-p)
    peak=float(logs.max())
    return float(np.exp(peak)*np.exp(logs-peak).sum())

def clopper_pearson_table(trials,groups=16,alpha=.05):
    n=int(trials);tail=float(alpha)/(2*groups)
    bounds=np.empty((n+1,2),float)
    for count in range(n+1):
        if count==0:lower=0.
        elif count==n:lower=tail**(1/n)
        else:
            left,right=0.,1.
            for _ in range(90):
                mid=(left+right)/2
                if binomial_cdf(count-1,n,mid)>1-tail:left=mid
                else:right=mid
            lower=(left+right)/2
        if count==n:upper=1.
        elif count==0:upper=1-tail**(1/n)
        else:
            left,right=0.,1.
            for _ in range(90):
                mid=(left+right)/2
                if binomial_cdf(count,n,mid)>tail:left=mid
                else:right=mid
            upper=(left+right)/2
        bounds[count]=lower,upper
    assert abs(bounds[0,1]-(1-tail**(1/n)))<2e-14
    assert abs(bounds[n,0]-tail**(1/n))<2e-14
    assert np.all(bounds[:,0]<=np.arange(n+1)/n)and np.all(bounds[:,1]>=np.arange(n+1)/n)
    assert np.all(np.diff(bounds[:,0])>=0)and np.all(np.diff(bounds[:,1])>=0)
    return bounds

def odds_shift(probability,shift):
    p=np.asarray(probability,float)
    factor=np.exp(np.asarray(shift,float))
    return factor*p/(1-p+factor*p)

def estimate_from_counts(reference_group,counts,trials,base_bounds,bias_logodds,width,concentration):
    """Target and uncertainty use only reference, finite counts, and settings.

    The first half of groups are the declared rare-reference groups.  A fixed
    prefix receives the declared log-odds bias.  No true probability or future
    evaluation outcome is accepted by this function.
    """
    b=np.asarray(reference_group,float);x=np.asarray(counts,int)
    affected=int(round(len(b)/2*concentration))
    shift=np.zeros(len(b));shift[:affected]=bias_logodds
    t=odds_shift(x/trials,shift)
    nominal_low=odds_shift(base_bounds[x,0],shift)
    nominal_high=odds_shift(base_bounds[x,1],shift)
    lower=np.maximum(0,t-width*(t-nominal_low))
    upper=np.minimum(1,t+width*(nominal_high-t))
    assert np.all(lower<=t+1e-15)and np.all(upper>=t-1e-15)
    return t,lower,upper,shift
