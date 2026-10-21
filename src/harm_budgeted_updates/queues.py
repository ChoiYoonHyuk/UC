"""Evaluate a fixed-capacity review queue on the full deployment cohort.

Unknown responses remain unknown. Known-positive recall has the number of
known positive responses in the cohort as its denominator, while queue
capacity is computed using every deployment case. IDs break score ties.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.special import logit


def evaluate_queue(b,p,y,fraction=.01,ids=None,margin_cutpoints=(.25,1.)):
    """Return retrospective queue outcomes and baseline-margin strata.

    Responses are 0, 1, or a negative value indicating an unknown response.
    Inputs are aligned vectors. A unique ID gives stable ordering under ties.
    The score maps are supplied by the caller and are never fitted here.
    """
    b,p=np.asarray(b,float),np.asarray(p,float)
    y=np.asarray(y)
    if b.ndim!=1 or b.shape!=p.shape or b.shape!=y.shape or len(b)==0:
        raise ValueError('b, p, and y must be nonempty aligned vectors')
    if not 0<fraction<=1:raise ValueError('fraction must belong to (0, 1]')
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(p)):
        raise ValueError('Scores must be finite')
    if np.any((b<=0)|(b>=1)|(p<=0)|(p>=1)):
        raise ValueError('Scores must be strictly between zero and one')
    if not np.all((y==0)|(y==1)|(y<0)):
        raise ValueError('Responses must be 0, 1, or negative for unknown')
    ids=np.arange(len(b)) if ids is None else np.asarray(ids)
    if ids.shape!=b.shape or len(np.unique(ids))!=len(ids):
        raise ValueError('IDs must be unique and aligned with scores')
    # Sorting IDs first reproduces the stable score order of the original study.
    canonical=np.argsort(ids,kind='stable')
    b,p,y=b[canonical],p[canonical],y[canonical]
    n=len(y);k=max(1,int(np.ceil(fraction*n)))
    order=np.argsort(-b,kind='stable')
    before=np.zeros(n,bool);before[order[:k]]=True
    after=np.zeros(n,bool);after[np.argsort(-p,kind='stable')[:k]]=True
    lost=before&~after;enter=after&~before;retain=before&after
    cutoff=b[order[k-1]]
    gap=np.maximum(0,logit(b)-logit(cutoff))
    stratum=np.minimum(np.searchsorted(margin_cutpoints,gap,side='right'),2)
    result=dict(queue_fraction=float(fraction),n_deployment=n,capacity=k,
        reference_cutoff_probability=float(cutoff),reference_cutoff_logit=float(logit(cutoff)),
        reference_known_TP=int(np.sum(before&(y==1))),reference_known_FP=int(np.sum(before&(y==0))),
        reference_unknown=int(np.sum(before&(y<0))),known_TP=int(np.sum(after&(y==1))),
        known_FP=int(np.sum(after&(y==0))),unknown=int(np.sum(after&(y<0))),
        all_lost=int(lost.sum()),all_enter=int(enter.sum()),
        known_lost_TP=int(np.sum(lost&(y==1))),known_enter_TP=int(np.sum(enter&(y==1))),
        known_lost_FP=int(np.sum(lost&(y==0))),known_enter_FP=int(np.sum(enter&(y==0))),
        unknown_lost=int(np.sum(lost&(y<0))),unknown_enter=int(np.sum(enter&(y<0))),
        retained_known_TP=int(np.sum(retain&(y==1))),retained_known_FP=int(np.sum(retain&(y==0))),
        retained_unknown=int(np.sum(retain&(y<0))))
    for field in ['all_lost','all_enter','known_lost_TP','known_enter_TP','known_lost_FP','known_enter_FP',
                  'unknown_lost','unknown_enter','retained_known_TP','retained_known_FP','retained_unknown',
                  'known_TP','known_FP','unknown']:
        result[field+'_per_capacity']=result[field]/k
    result['precision_identification_lower']=result['known_TP']/k
    result['precision_identification_upper']=(result['known_TP']+result['unknown'])/k
    positives=int(np.sum(y==1))
    result['known_positive_total']=positives
    result['known_positive_recall_denominator_zero']=int(positives==0)
    result['known_positive_recall']=None if positives==0 else result['known_TP']/positives
    assert result['known_TP']+result['known_FP']+result['unknown']==k
    assert result['all_lost']==result['all_enter']
    for cls,label in [(1,'known_TP'),(0,'known_FP'),(-1,'unknown')]:
        member=y==cls if cls>=0 else y<0
        assert result['reference_'+label]-int(np.sum(lost&member))+int(np.sum(enter&member))==result[label]
    margins=[]
    for j,name in enumerate(['near','middle','far']):
        member=before&(stratum==j);positive=member&(y==1)
        count=int(member.sum());pos=int(positive.sum())
        a=int(np.sum(lost&member));v=int(np.sum(lost&positive))
        margins.append(dict(margin_stratum=name,capacity=k,reference_members=count,reference_known_TP=pos,
            departures=a,known_positive_departures=v,departures_per_capacity=a/k,
            known_positive_departures_per_capacity=v/k,
            departure_fraction_of_stratum=None if count==0 else a/count,
            known_positive_departure_fraction=None if pos==0 else v/pos,
            zero_member_denominator=int(count==0),zero_known_positive_denominator=int(pos==0)))
    return result,margins


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='NPZ containing b, p, y, and optional ids.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fractions',type=float,nargs='+',default=[.005,.01,.02])
    args=parser.parse_args()
    with np.load(args.input,allow_pickle=False) as z:
        rows=[dict(metrics=r,margins=m) for r,m in [evaluate_queue(z['b'],z['p'],z['y'],f,z['ids'] if 'ids' in z else None) for f in args.fractions]]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as stream:json.dump(rows,stream,indent=2,allow_nan=False)


if __name__=='__main__':cli()
