"""A sealed dependent-response illustration of label-uniform harm budgets.

The Hoeffding reference is valid under independent responses with known mean.
No real graph or fitted fraud prevalence is used in this mechanism experiment.
"""
from pathlib import Path
import csv,hashlib,itertools,json,math,time
import numpy as np
from .allocation import allocate_binary,dinfinity_binary
from .allocation import uniform_cap_binary
from .allocation import robust_common_binary,worst_excess_loss
ROOT=None
ALPHA=.05;H=.0025;TAU=.1;ETA=.1;REPS=16384
RINGS=[1,4,16,64,256,512];RHOS=[0.,.25,.5,.75,1.]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf8')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def profile(n):
    i=np.arange(n);b=.2+.2*(i+.5)/n;t=.04+.02*(1+np.cos(2*np.pi*i/n));eta=np.full(n,ETA);return b,t,eta
def allocate(n):
    b,t,eta=profile(n);lo=np.maximum(0,t-.5*np.minimum(b,1-b));hi=np.minimum(1,t+.5*np.minimum(b,1-b))
    uc,ur=uniform_cap_binary(b,t,H,TAU);point,pr=allocate_binary(b,t,H,TAU);ruc,rr=robust_common_binary(b,lo,hi,H,comparator=uc,tau=TAU)
    policies=dict(UC=uc,Point=point,RUC=ruc);meta=[]
    for name,p in policies.items():
        c=dinfinity_binary(b,p);d1=np.maximum(np.log(b/p),0);d0=np.maximum(np.log((1-b)/(1-p)),0);mu=float(np.mean(eta*d1+(1-eta)*d0));raw=mu+math.sqrt(float(c@c)*math.log(1/ALPHA)/(2*n*n));bound=min(float(c.mean()),raw)
        assert np.all(p<=b+1e-14)and np.all((eta>=lo)&(eta<=hi));assert c.mean()<=H+2e-13 and c.max()<=TAU+2e-13
        risk=float(np.mean(-eta*np.log(p)-(1-eta)*np.log1p(-p)))
        meta.append(dict(n=n,policy=name,mean_cost=float(c.mean()),maximum_cost=float(c.max()),expected_harm=mu,conditional_nll=risk,hoeffding_raw_upper=raw,hoeffding_upper=bound,alpha=ALPHA,active_fraction=float(np.mean(c>1e-12)),sum_squared_cost=float(c@c),prediction_sha256=ah(p)))
    assert np.mean(worst_excess_loss(ruc,uc,lo,hi))<=2e-13
    assert meta[2]['conditional_nll']<=meta[0]['conditional_nll']+2e-13
    return dict(b=b,t=t,eta=eta,lower=lo,upper=hi,**policies),meta,dict(UC=ur,Point=pr,RUC=rr)
def graph(n,g):
    edges=[]
    for start in range(0,n,g):
        if g==1:continue
        if g==2:edges.append((start,start+1));continue
        edges.extend((start+j,start+(j+1)%g)for j in range(g))
    return np.array(edges,dtype=np.int64).reshape(-1,2)
def wilson(k,n):
    z=1.959963984540054;v=k/n;den=1+z*z/n;mid=(v+z*z/(2*n))/den;half=z*math.sqrt(v*(1-v)/n+z*z/(4*n*n))/den;return mid-half,mid+half
def harm_vectors(y,data,p):
    b=data['b'];h1=np.maximum(np.log(b/p),0);h0=np.maximum(np.log((1-b)/(1-p)),0);return np.where(y,h1,h0)
def main():
    start=time.time();ROOT.mkdir(exist_ok=True)
    protocol=dict(version='v23_ring_dependence_1',n=512,eta=ETA,h=H,tau=TAU,interval_kappa=.5,ring_sizes=RINGS,rhos=RHOS,alpha=ALPHA,joint_repetitions=REPS,
        profile='b_i=.2+.2(i+.5)/n; t_i=.04+.02(1+cos(2*pi*i/n)); i=0,...,n-1; eta_i=.1. Identical across every dependence condition.',
        graph='Disjoint components: isolated vertices at size1; one undirected edge at size2; simple cycles for size>=3. Consecutive vertex IDs form each component.',
        law='Each component independently chooses a shared-threshold branch with probability rho; that branch uses one Uniform(0,1) threshold for all component Bernoulli(eta) responses. Otherwise each vertex uses an independent Uniform threshold. All marginal probabilities remain eta=.1, within-component pairwise Pearson correlation equals rho when size>=2, and different components are independent.',
        application_scope='Mechanism illustration of dependent fraud-activity groups, not a fitted fraud model or empirical fraud-prevalence estimate.',
        methods=['UC','Point','RUC'],allocation='Computed once per cohort size. RUC uses original feasible UC(h) as fixed comparator. No map or bound is refitted as dependence changes.',
        benchmark='If responses are independent, harmful losses lie in [0,c_i] and have exact mean mu. Hoeffding gives P(H>mu+sqrt(log(1/alpha)*sum(c_i^2)/(2*n^2)))<=alpha. The reported upper is also clipped by deterministic C. This is an explicit independence-assumption benchmark, not attributed to any generic graph algorithm.',
        exact='Enumerate all2^8 responses at n8 for component sizes1,2,4,8 and allrho; use exact component-mixture probabilities. At n512,rho1, enumerate all component-constant patterns for sizes64,256,512 (at most8components).',
        outcomes='Exact marginal NLL and E[H]; Monte Carlo joint distributions of mean harm and within-vector top1% harm; bound exceedance with Wilson intervals; deterministic violations; exact response-vector checks and exact limiting-law tail/probability values.',
        random_state=dict(generator='PCG64',state=2026102013),common_random_numbers='One independent vertex-uniform array shared over allcells; each ring size has independent component-threshold and branch-uniform arrays reused over allrho. Singleton threshold equals its independent vertex threshold.',
        complete_reporting=True,source_sha256={Path(__file__).name:sha(__file__)})
    pp=ROOT/'protocol_v23.json';assert not pp.exists(),'Preserve completed sealed run';pp.write_text(json.dumps(protocol,indent=2)+'\n')
    (ROOT/'protocol_seal_v23.json').write_text(json.dumps(dict(protocol_sha256=sha(pp),unix_time=time.time(),before_any_allocation_or_responses=True),indent=2)+'\n')
    saved={};profiles=[];solvers={}
    for n in [8,512]:
        data,meta,record=allocate(n);saved.update({f'n{n}_{k}':v for k,v in data.items()});profiles.extend(meta);solvers[str(n)]=record
    np.savez_compressed(ROOT/'probabilities_v23.npz',**saved);csvwrite(ROOT/'policy_profiles_v23.csv',profiles)
    (ROOT/'probability_commit_v23.json').write_text(json.dumps(dict(complete=True,protocol_sha256=sha(pp),probabilities_sha256=sha(ROOT/'probabilities_v23.npz'),profiles_sha256=sha(ROOT/'policy_profiles_v23.csv'),solver_records=solvers,committed_before_joint_responses=True),indent=2)+'\n')
    data={k:saved['n512_'+k]for k in ['b','t','eta','lower','upper','UC','Point','RUC']};n=512;k=math.ceil(.01*n);rng=np.random.default_rng(protocol['random_state']['state']);ind=rng.random((REPS,n));packed={};rows=[];distribution={};edges={}
    for g in RINGS:
        G=n//g;shared=ind if g==1 else rng.random((REPS,G));branch=rng.random((REPS,G));edges[f'g{g}']=graph(n,g)
        for rho in RHOS:
            y=np.where(np.repeat(branch<rho,g,axis=1),np.repeat(shared<ETA,g,axis=1),ind<ETA)
            key=f'g{g}_rho{rho:g}';packed[key]=np.packbits(y,axis=1)
            for name in protocol['methods']:
                p=data[name];m=next(v for v in profiles if v['n']==n and v['policy']==name);harm=harm_vectors(y,data,p);mean=harm.mean(axis=1);tail=np.partition(harm,n-k,axis=1)[:,-k:].mean(axis=1);bound=m['hoeffding_upper'];exceed=mean>bound+1e-14;vio=mean>m['mean_cost']+1e-13;capvio=np.any(harm>TAU+1e-13,axis=1);assert not vio.any()and not capvio.any();low,high=wilson(int(exceed.sum()),REPS)
                distribution[key+'_'+name+'_H']=mean;distribution[key+'_'+name+'_T1']=tail
                rows.append(dict(n=n,ring_size=g,rho=rho,components=G,policy=name,joint_draws=REPS,mean_cost=m['mean_cost'],maximum_cost=m['maximum_cost'],exact_expected_H=m['expected_harm'],conditional_nll=m['conditional_nll'],hoeffding_upper=bound,hoeffding_raw_upper=m['hoeffding_raw_upper'],alpha=ALPHA,mean_H=float(mean.mean()),mean_H_mcse=float(mean.std(ddof=1)/math.sqrt(REPS)),H_q50=float(np.quantile(mean,.5)),H_q90=float(np.quantile(mean,.9)),H_q95=float(np.quantile(mean,.95)),H_q99=float(np.quantile(mean,.99)),H_maximum=float(mean.max()),expected_T1=float(tail.mean()),T1_mcse=float(tail.std(ddof=1)/math.sqrt(REPS)),T1_q95=float(np.quantile(tail,.95)),T1_maximum=float(tail.max()),independence_bound_exceedance=float(exceed.mean()),exceedance_low=low,exceedance_high=high,deterministic_violations=int(vio.sum()),individual_cap_violations=int(capvio.sum())))
            print(json.dumps(dict(ring_size=g,rho=rho,seconds=round(time.time()-start,2))),flush=True)
    np.savez_compressed(ROOT/'response_vectors_v23.npz',**packed);np.savez_compressed(ROOT/'joint_statistics_v23.npz',**distribution);np.savez_compressed(ROOT/'ring_graphs_v23.npz',**edges);csvwrite(ROOT/'dependence_metrics_v23.csv',rows)
    exact=[]
    for n in [8,512]:
        data={k:saved[f'n{n}_'+k]for k in ['b','t','eta','UC','Point','RUC']};k=math.ceil(.01*n)
        for g in ([1,2,4,8]if n==8 else[64,256,512]):
            for rho in(RHOS if n==8 else[1.]):
                if n==8:
                    y=np.array(list(itertools.product([0,1],repeat=n)),np.uint8);counts=y.reshape(-1,n//g,g).sum(axis=2);prob=np.prod((1-rho)*(ETA**counts)*(1-ETA)**(g-counts)+rho*((counts==0)*(1-ETA)+(counts==g)*ETA),axis=1)
                else:
                    patterns=np.array(list(itertools.product([0,1],repeat=n//g)),np.uint8);y=np.repeat(patterns,g,axis=1);counts=patterns.sum(axis=1);prob=ETA**counts*(1-ETA)**(n//g-counts)
                assert abs(prob.sum()-1)<2e-14;assert np.max(abs(prob@y-ETA))<2e-14
                for name in protocol['methods']:
                    m=next(v for v in profiles if v['n']==n and v['policy']==name);harm=harm_vectors(y,data,data[name]);mean=harm.mean(axis=1);tail=np.sort(harm,axis=1)[:,-k:].mean(axis=1);assert np.max(mean)<=m['mean_cost']+1e-13;failure=float(prob@(mean>m['hoeffding_upper']+1e-14));mu=float(prob@mean);assert abs(mu-m['expected_harm'])<2e-14
                    if n==512 and g==512 and m['hoeffding_upper']<m['mean_cost']-1e-12:assert abs(failure-ETA)<2e-14
                    exact.append(dict(n=n,ring_size=g,rho=rho,policy=name,response_vectors=len(y),positive_probability_vectors=int(np.sum(prob>0)),expected_H=mu,expected_T1=float(prob@tail),maximum_H=float(mean.max()),mean_cost=m['mean_cost'],hoeffding_upper=m['hoeffding_upper'],independence_bound_exceedance=failure,deterministic_violations=int(np.sum((mean>m['mean_cost']+1e-13)&(prob>0)))))
    csvwrite(ROOT/'exact_dependence_v23.csv',exact)
    receipt=dict(status='PASS',complete=True,probability_profiles=len(profiles),simulation_cells=len(rows),dependency_conditions=len(RINGS)*len(RHOS),joint_draws_per_condition=REPS,exact_cells=len(exact),protocol_sha256=sha(pp),probability_commit_sha256=sha(ROOT/'probability_commit_v23.json'),wall_seconds=time.time()-start,any_deterministic_violation=False)
    (ROOT/'run_receipt_v23.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
def cli():
    import argparse
    global ROOT, REPS, RINGS, RHOS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=16384)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    ROOT=args.output.resolve();ROOT.mkdir(parents=True,exist_ok=True)
    REPS=args.repetitions
    if args.smoke:REPS=32;RINGS=[1,512];RHOS=[0.,1.]
    if REPS<2:parser.error('--repetitions must be at least two')
    main()

if __name__=='__main__':cli()
