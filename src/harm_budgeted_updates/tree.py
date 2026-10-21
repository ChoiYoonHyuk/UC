"""Exact tree-model experiment with analytical omitted-interaction enclosures.

The complete protocol is written before any realization is generated.  Exact
conditional marginals are diagnostics only and are not allocator inputs.
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import time
import numpy as np
from .allocation import (allocate_binary,common_strength_binary,
    floor_bounds,dinfinity_binary,binary_cross_entropy)
from .allocation import robust_common_binary,worst_excess_loss

ROOT=None
DRAWS=30
N=63
STRENGTHS=[.025,.075,.2]
DENSITIES=[.05,.2,.5]
RADII=[0,2,10]
SPINS=np.array([-1.,1.])
H=.0025
TAU=.1

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sigmoid(x):
    return 1/(1+np.exp(-np.asarray(x)))

def logit(x):
    return np.log(x)-np.log1p(-x)

def lse(x):
    maximum=np.max(x)
    return maximum+np.log(np.exp(x-maximum).sum())

def exact_tree(fields,edges,couplings,observed,labels):
    """Return exact marginals and directed log messages for a binary tree."""
    n=len(fields)
    neighbors=[[] for _ in range(n)]
    for (u,v),j in zip(edges,couplings):
        neighbors[int(u)].append((int(v),j))
        neighbors[int(v)].append((int(u),j))
    messages={}
    def local(u):
        score=fields[u]*SPINS
        if observed[u]:
            score=score.copy()
            score[int(1-labels[u])]=-np.inf
        return score
    def message(u,v):
        key=(u,v)
        if key in messages:
            return messages[key]
        score=local(u).copy()
        j_uv=0.
        for w,j in neighbors[u]:
            if w==v:
                j_uv=j
            else:
                score+=message(w,u)
        result=np.array([lse(score+j_uv*SPINS*s) for s in SPINS])
        result-=lse(result)
        messages[key]=result
        return result
    marginals=np.zeros(n)
    for u in range(n):
        score=local(u).copy()
        for v,j in neighbors[u]:
            score+=message(v,u)
        marginals[u]=np.exp(score[1]-lse(score))
    return marginals,messages,neighbors

def sample_tree(fields,edges,couplings,rng):
    n=len(fields)
    marginal,messages,neighbors=exact_tree(fields,edges,couplings,np.zeros(n,bool),np.zeros(n,int))
    labels=np.zeros(n,int)
    labels[0]=rng.random()<marginal[0]
    def descend(u,parent):
        for v,j in neighbors[u]:
            if v==parent:
                continue
            score=fields[v]*SPINS+j*SPINS*SPINS[labels[u]]
            for w,j2 in neighbors[v]:
                if w!=u:
                    score+=messages[(w,v)]
            prob=np.exp(score[1]-lse(score))
            labels[v]=rng.random()<prob
            descend(v,u)
    descend(0,-1)
    return labels

def exact_enumeration(fields,edges,couplings,observed,labels):
    free=np.flatnonzero(~observed)
    configurations=np.tile(labels,(2**len(free),1))
    configurations[:,free]=np.array(list(itertools.product([0,1],repeat=len(free))))
    x=2*configurations-1
    logw=x@fields+(x[:,edges[:,0]]*x[:,edges[:,1]])@couplings
    w=np.exp(logw-lse(logw))
    return w@configurations

def regional_marginals(fields,edges,couplings,observed,labels,radius):
    n=len(fields)
    _,_,neighbors=exact_tree(fields,edges,couplings,observed,labels)
    target=np.full(n,np.nan)
    gamma=np.full(n,np.nan)
    cut_matrix=np.zeros((n,len(edges)),bool)
    for center in np.flatnonzero(~observed):
        region={int(center)}
        frontier={int(center)}
        for distance in range(radius):
            frontier={v for u in frontier for v,j in neighbors[u]
                      if not observed[v] and v not in region}
            region.update(frontier)
            if not frontier:
                break
        inside=np.array([u in region for u in range(n)])
        cut=(inside[edges[:,0]]!=inside[edges[:,1]])&(~observed[edges[:,0]])&(~observed[edges[:,1]])
        local_couplings=couplings.copy()
        local_couplings[cut]=0.
        local,_,_=exact_tree(fields,edges,local_couplings,observed,labels)
        target[center]=local[center]
        gamma[center]=2*np.abs(couplings[cut]).sum()
        cut_matrix[center]=cut
    return target,gamma,cut_matrix

def uniform_cap(b,t,h):
    def at(tau):
        a,d=floor_bounds(b,tau)
        return np.clip(t,a,d)
    p=at(TAU)
    if dinfinity_binary(b,p).mean()<=h:
        return p
    lo,hi=0.,TAU
    for _ in range(60):
        mid=(lo+hi)/2
        if dinfinity_binary(b,at(mid)).mean()<=h:
            lo=mid
        else:
            hi=mid
    return at(lo)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    start=time.time()
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'native').mkdir(exist_ok=True)
    protocol=dict(version='analytical_tree_v21_1',draws=DRAWS,n=N,
        graph=f'balanced binary tree, vertices 0 through {N-1} and parent floor((v-1)/2)',
        fields='independent Normal(-.65,.9^2)',
        old_fields='0.4*true_field + independent Normal(.15,.45^2)',
        reference='sigmoid(2*old_field), before any observed response',
        interaction_strengths=STRENGTHS,
        interaction='strength * Uniform(.75,1.25) * sign; sign -1 with probability .6',
        observed_densities=DENSITIES,observed_selection='nested label-independent permutation',
        local_radii=RADII,complete_radius=max(RADII),
        target='exact marginal after removal of unobserved region boundary edges',
        retained_information='all local fields and all observed responses; observed-neighbor interactions retained',
        omitted_oscillation='Gamma=2 sum(abs(J_e)) over removed boundary edges on common clamped support',
        gamma_multipliers=[0.,.25,1.],primary_gamma_multiplier=1.,
        primary_comparator='Common',allowance=H,per_case_cap=TAU,
        controls=['Input','Common','Point','UC','M.25','M.5','Robust'],
        methods_comparison='same permitted upper bound; actual costs reported without forced expenditure',
        batches='only unobserved vertices; observed deterministic responses excluded',
        conditional_risk='exact full-tree conditional eta, never an optimization input',
        independence=f'{DRAWS} independent field/coupling/sign/observation-order realizations; all factorial conditions paired within draw',
        bootstrap='10000 paired draw-block replicates, factorial settings averaged within draw',
        complete_case_reporting=True,
        source_hashes={Path(__file__).name:sha(__file__)})
    protocol_path=ROOT/'protocol_v21.json'
    if protocol_path.exists():
        raise RuntimeError('Immutable protocol already exists; use saved experiment outputs')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
    (ROOT/'protocol_seal_v21.json').write_text(json.dumps(dict(sha256=sha(protocol_path),sealed_before_any_model_draw=True,unix_time=time.time()),indent=2)+'\n')
    rng=np.random.default_rng(2026102033)
    # Independent verification before applying the implementation to the study.
    enumeration_checks=[]
    for trial in range(12):
        n=10
        edges=np.array([[(v-1)//2,v] for v in range(1,n)],int)
        fields=rng.normal(-.3,.8,n)
        j=rng.uniform(-.4,.4,len(edges))
        obs=rng.random(n)<.25
        labels=rng.integers(0,2,n)
        exact,_,_=exact_tree(fields,edges,j,obs,labels)
        enum=exact_enumeration(fields,edges,j,obs,labels)
        error=float(np.max(np.abs(exact-enum)))
        assert error<3e-14,error
        enumeration_checks.append(dict(trial=trial,vertices=n,observed=int(obs.sum()),max_error=error))
    edges=np.array([[(v-1)//2,v] for v in range(1,N)],int)
    metrics=[]
    conditions=[]
    vertices=[]
    audit=dict(max_containment_error=0.,max_risk_bound_error=0.,max_directional_order_error=0.,max_budget_violation=0.,max_cap_violation=0.)
    for draw in range(DRAWS):
        fields=rng.normal(-.65,.9,N)
        old_fields=.4*fields+rng.normal(.15,.45,N)
        b_all=sigmoid(2*old_fields)
        j_unit=rng.uniform(.75,1.25,len(edges))*np.where(rng.random(len(edges))<.6,-1.,1.)
        observation_order=rng.permutation(N)
        native=dict(fields=fields,old_fields=old_fields,b=b_all,edges=edges,j_unit=j_unit,observation_order=observation_order)
        for strength in protocol['interaction_strengths']:
            j=strength*j_unit
            labels=sample_tree(fields,edges,j,rng)
            for density in protocol['observed_densities']:
                observed=np.zeros(N,bool)
                observed[observation_order[:round(N*density)]]=True
                use=~observed
                eta_all,_,_=exact_tree(fields,edges,j,observed,labels)
                eta=eta_all[use]
                b=b_all[use]
                for radius in protocol['local_radii']:
                    name=f'd{draw:02d}_j{strength:g}_o{density:g}_r{radius}'
                    t_all,gamma_all,cut=regional_marginals(fields,edges,j,observed,labels,radius)
                    t=t_all[use]
                    gamma=gamma_all[use]
                    q,_=common_strength_binary(b,t,H,TAU)
                    point,_=allocate_binary(b,t,H,TAU)
                    uc=uniform_cap(b,t,H)
                    basic={'Input':b,'Common':q,'Point':point,'UC':uc,
                           'M.25':.75*q+.25*point,'M.5':.5*q+.5*point}
                    saved={**native,'couplings':j,'labels':labels,'observed':observed,
                           'eta':eta_all,'target':t_all,'gamma':gamma_all,'cut_matrix':cut}
                    for multiplier in protocol['gamma_multipliers']:
                        lower=sigmoid(logit(t)-multiplier*gamma)
                        upper=sigmoid(logit(t)+multiplier*gamma)
                        robust,record=robust_common_binary(b,lower,upper,H,comparator=q,tau=TAU)
                        policies={**basic,'Robust':robust}
                        departure=logit(robust)-logit(q)
                        miss=np.maximum(np.maximum(lower-eta,eta-upper),0.)
                        xi=float(np.mean(miss*np.abs(departure)))
                        xi_dir=float(np.mean(np.maximum(lower-eta,0.)*np.maximum(departure,0.)+np.maximum(eta-upper,0.)*np.maximum(-departure,0.)))
                        vr=float(worst_excess_loss(robust,q,lower,upper).mean())
                        riskdiff=float(np.mean(-(eta*np.log(robust/q)+(1-eta)*np.log((1-robust)/(1-q)))))
                        margin=float(max(np.max(lower-eta),np.max(eta-upper)))
                        if multiplier==1.:
                            assert margin<3e-14,(name,margin)
                            assert riskdiff<=3e-14,(name,riskdiff)
                            audit['max_containment_error']=max(audit['max_containment_error'],margin)
                        assert riskdiff<=vr+xi_dir+4e-14,(name,riskdiff,vr,xi_dir)
                        assert xi_dir<=xi+4e-14
                        audit['max_risk_bound_error']=max(audit['max_risk_bound_error'],riskdiff-vr-xi_dir)
                        audit['max_directional_order_error']=max(audit['max_directional_order_error'],xi_dir-xi)
                        common_risk=binary_cross_entropy(eta,q)
                        point_risk=binary_cross_entropy(eta,point)
                        condition=dict(draw=draw,strength=strength,observed_density=density,radius=radius,
                            gamma_multiplier=multiplier,n=len(b),mean_gamma=float(gamma.mean()),
                            mean_width=float((upper-lower).mean()),simultaneous_containment=float(margin<=3e-14),
                            vertex_containment=float(np.mean((eta>=lower-3e-14)&(eta<=upper+3e-14))),
                            maximum_interval_error=max(0.,margin),comparator_identity=float(np.max(np.abs(robust-q))<1e-12),
                            nonidentity_vertex_fraction=float(np.mean(np.abs(robust-q)>=1e-12)),
                            mean_logit_departure=float(np.mean(np.abs(departure))),
                            robust_value=vr,xi_u=xi,xi_dir=xi_dir,upper_u=vr+xi,upper_dir=vr+xi_dir,
                            symmetric_certificate=float(vr+xi<=1e-12),directional_certificate=float(vr+xi_dir<=1e-12),
                            riskdiff_common=riskdiff,riskdiff_point=binary_cross_entropy(eta,robust)-point_risk,
                            common_risk=common_risk,point_risk=point_risk,
                            conditional_improvement=float(riskdiff<-1e-12),
                            point_conditional_worse=float(point_risk>common_risk+1e-12),
                            robust_positive_risk=float(riskdiff>1e-12))
                        conditions.append(condition)
                        for policy,p in policies.items():
                            cost=dinfinity_binary(b,p)
                            assert cost.mean()<=H+3e-14 and cost.max()<=TAU+3e-14
                            audit['max_budget_violation']=max(audit['max_budget_violation'],float(cost.mean()-H))
                            audit['max_cap_violation']=max(audit['max_cap_violation'],float(cost.max()-TAU))
                            harm=np.maximum(-(labels[use]*np.log(p/b)+(1-labels[use])*np.log((1-p)/(1-b))),0.)
                            expected_harm=eta*np.maximum(np.log(b/p),0.)+(1-eta)*np.maximum(np.log((1-b)/(1-p)),0.)
                            metrics.append(dict(draw=draw,strength=strength,observed_density=density,radius=radius,
                                gamma_multiplier=multiplier,method=policy,n=len(b),
                                conditional_risk=binary_cross_entropy(eta,p),riskdiff_common=binary_cross_entropy(eta,p)-common_risk,
                                riskdiff_input=binary_cross_entropy(eta,p)-binary_cross_entropy(eta,b),
                                mean_cost=float(cost.mean()),maximum_cost=float(cost.max()),
                                expected_harm=float(expected_harm.mean()),realized_harm=float(harm.mean()),
                                tail_harm=float(np.sort(harm)[-max(1,int(np.ceil(len(harm)*.01))):].mean()),
                                changed_input=float(np.mean(np.abs(p-b)>1e-12)),
                                equal_common=float(np.mean(np.abs(p-q)<=1e-12))))
                            saved[f'p_{policy.replace(".","_")}_g{multiplier:g}']=p
                        saved[f'L_g{multiplier:g}']=lower
                        saved[f'U_g{multiplier:g}']=upper
                        if multiplier==1.:
                            for k,vertex in enumerate(np.flatnonzero(use)):
                                vertices.append(dict(draw=draw,strength=strength,observed_density=density,radius=radius,
                                    vertex=int(vertex),b=b[k],target=t[k],eta=eta[k],gamma=gamma[k],lower=lower[k],upper=upper[k],
                                    common=q[k],robust=robust[k],point=point[k],observed=int(observed[vertex]),
                                    omitted_edges=int(cut[vertex].sum())))
                    np.savez_compressed(ROOT/'native'/f'{name}.npz',**saved)
        print(json.dumps(dict(draw=draw,conditions=len(conditions),elapsed=round(time.time()-start,2))),flush=True)
    write_csv(ROOT/'results'/'conditions_v21.csv',conditions)
    write_csv(ROOT/'results'/'method_metrics_v21.csv',metrics)
    write_csv(ROOT/'results'/'vertex_enclosures_v21.csv',vertices)
    receipt=dict(status='PASS',complete=True,protocol_sha256=sha(protocol_path),draws=DRAWS,
        condition_count=len(conditions),primary_condition_count=sum(v['gamma_multiplier']==1. for v in conditions),
        method_rows=len(metrics),vertex_rows=len(vertices),direct_enumeration_checks=enumeration_checks,
        inequality_checks=audit,native_files=len(list((ROOT/'native').glob('*.npz'))),
        wall_seconds=time.time()-start)
    (ROOT/'run_audit_v21.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2),flush=True)

def cli():
    import argparse
    global ROOT, DRAWS, N, STRENGTHS, DENSITIES, RADII
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=30)
    parser.add_argument('--smoke',action='store_true',help='Run one small factorial condition, preserving the original generator.')
    args=parser.parse_args()
    ROOT=args.output.resolve();ROOT.mkdir(parents=True,exist_ok=True)
    DRAWS=args.draws
    if args.smoke:
        DRAWS=1;N=15;STRENGTHS=[.075];DENSITIES=[.2];RADII=[2]
    if DRAWS<1:parser.error('--draws must be positive')
    main()

if __name__=='__main__':cli()
