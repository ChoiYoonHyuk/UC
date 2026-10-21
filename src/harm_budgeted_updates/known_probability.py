"""Complete inherited finite-source and conditional-tree mechanism study.

Protocol is immutable before allocation. All deployable maps are committed
before new joint evaluation responses are generated. Exact eta is used only
for diagnostic width-placement controls and the explicitly marked oracle.
"""
from pathlib import Path
import csv, hashlib, json, time, itertools
import numpy as np
from .allocation import allocate_binary, common_strength_binary, floor_bounds, dinfinity_binary
from .allocation import robust_common_binary, worst_excess_loss
from .allocation import uniform_cap_binary
from .tree import exact_tree

ROOT=None
FINITE=None
TREE=None
DRAWS=20
JOINT=512
TAU=.1

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def logit(p):return np.log(p)-np.log1p(-p)
def ce(eta,p):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(eta==0.,0.,-eta*np.log(p))+np.where(eta==1.,0.,-(1-eta)*np.log1p(-p))
def write(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def seal():
    sources={Path(__file__).name:sha(__file__)}
    protocol=dict(version='known_probability_v22_1',created_unix=time.time(),
      finite=dict(laws=3*DRAWS,rarities=[1e-4,1e-3,1e-2],draws_per_rarity=DRAWS,bias_factors=[.125,.5,1.,2.,8.],widths=[.5,1.,2.],affected_fractions=[.25,1.],conditions=90*DRAWS,allowances=[.0025,.01],batch_size=256,source_hashes={n:sha(FINITE/n) for n in ['finite_source_inputs_v18.npz','finite_source_targets_intervals_v18.npz']}),
      tree=dict(realizations=len({p.name.split('_')[0] for p in TREE.glob('*.npz')}),gamma_multipliers=[0.,.25,1.],batches=len(list(TREE.glob('*.npz'))),conditions=3*len(list(TREE.glob('*.npz'))),allowance=.0025,native_payloads={p.name:sha(p) for p in sorted(TREE.glob('*.npz'))}),
      cap=TAU,mixture_weights=[.25,.5,.75],comparators=['Common','UC'],
      placements=['Original','Aligned','Shuffled','Reversed'],
      controlled_width_rule='Take original effective widths U-L; make symmetric halfwidth templates. Aligned sorts templates in increasing |eta-t| rank, Reversed decreasing rank, Shuffled uses a deterministic law-specific permutation. Solve a positive scalar so that mean(min(1,t+s*w/2)-max(0,t-s*w/2)) equals original mean(U-L). Finite assignments are by sixteen source groups and retain sixteen equal vertices per group. Tree assignments are by unobserved vertex. All original asymmetric intervals remain separately reported.',
      eta_scope='Only controlled placement ordering and feasible oracle, never Original/UC/mixture targets.',
      methods=['Input','Native','Cap','Common','UC','Point','PC.25','PC.5','PC.75','PU.25','PU.5','PU.75','Oracle','RCommon','RUC'],
      joint_response_replicates=JOINT,joint_response_scope=f'{JOINT} response vectors shared across all policies and paired conditions with the same conditional law. Finite conditional responses independent across deployment vertices; conditional tree responses jointly sampled with all observed responses clamped.',
      expected_tail='For every response vector compute top ceil(.01*n) positive loss increases relative to b, then average; report Monte Carlo standard error and paired differences.',
      uncertainty='10000 paired independent-realization bootstrap draws, all factorial settings moved as one block; finite rarity averaged within each of twenty source draws, tree within thirty graph realizations.',
      complete_reporting=True,tail_optimization_extension=False,source_hashes=sources)
    out=ROOT/'protocol_v22.json'
    if out.exists():raise RuntimeError('Protocol already exists; preserve the completed run')
    out.write_text(json.dumps(protocol,indent=2)+'\n')
    (ROOT/'protocol_seal_v22.json').write_text(json.dumps(dict(protocol_sha256=sha(out),sealed_before_allocation=True,sealed_before_new_joint_responses=True,unix_time=time.time()),indent=2)+'\n')
    return protocol

def match_width(t, template, target):
    if target<=1e-16:return t.copy(),t.copy(),0.
    # A vanishing original width yields a vanishing template. Otherwise all
    # finite positive entries can reach the support boundaries under scaling.
    def at(s):return np.maximum(0.,t-s*template/2),np.minimum(1.,t+s*template/2)
    lo,hi=0.,1.
    while np.mean(at(hi)[1]-at(hi)[0])<target-2e-15:
        hi*=2
        if hi>2**60:raise ArithmeticError('Matched width unavailable')
    for _ in range(70):
        mid=(lo+hi)/2;l,u=at(mid)
        if np.mean(u-l)<target:lo=mid
        else:hi=mid
    l,u=at((lo+hi)/2)
    return l,u,(lo+hi)/2

def placements(t,eta,lower,upper,token,finite):
    yield 'Original',lower,upper,1.
    if finite:
        tt,ee,ww=t[::16],eta[::16],(upper-lower)[::16]
    else:tt,ee,ww=t,eta,upper-lower
    ranks=np.argsort(np.abs(ee-tt),kind='stable')
    ordered=np.sort(ww,kind='stable')
    rng=np.random.default_rng(int(hashlib.sha256(token.encode()).hexdigest()[:16],16))
    templates={}
    a=np.zeros_like(ww);a[ranks]=ordered;templates['Aligned']=a
    a=np.zeros_like(ww);a[ranks]=ordered[::-1];templates['Reversed']=a
    templates['Shuffled']=ww[rng.permutation(len(ww))]
    for name in ['Aligned','Shuffled','Reversed']:
        templ=np.repeat(templates[name],16) if finite else templates[name]
        # Structural zero original widths stay zero. If a subset of templates
        # is zero its theoretical maximum width can be smaller than target.
        limit=float(np.mean(templ>0))
        attainable=min(float(np.mean(upper-lower)),limit)
        l,u,scale=match_width(t,templ,attainable)
        yield name,l,u,scale

def controls(b,t,eta,h):
    point,_=allocate_binary(b,t,h,TAU)
    common,_=common_strength_binary(b,t,h,TAU)
    uc,_=uniform_cap_binary(b,t,h,TAU)
    oracle,_=allocate_binary(b,eta,h,TAU)
    lo,hi=floor_bounds(b,TAU)
    ans={'Input':b,'Native':t,'Cap':np.clip(t,lo,hi),'Common':common,'UC':uc,'Point':point,'Oracle':oracle}
    for alpha in [.25,.5,.75]:
        ans[f'PC{alpha:g}']=(1-alpha)*common+alpha*point
        ans[f'PU{alpha:g}']=(1-alpha)*uc+alpha*point
    return ans

def prediction_phase():
    start=time.time();meta=[]
    native=ROOT/'native';native.mkdir(exist_ok=True)
    finputs=np.load(FINITE/'finite_source_inputs_v18.npz')
    fsets=np.load(FINITE/'finite_source_targets_intervals_v18.npz')
    for ri,rarity in enumerate([1e-4,1e-3,1e-2]):
        for draw in range(DRAWS):
            base=f'r{ri}__d{draw:02d}';b=finputs[base+'__b'];eta=finputs[base+'__eta'];saved={'b':b,'eta':eta}
            for factor,width,affected in itertools.product([.125,.5,1.,2.,8.],[.5,1.,2.],[.25,1.]):
                key=base+f'__f{factor:g}__w{width:g}__c{affected:g}'
                t,l,u=[fsets[key+'__'+s] for s in ['target','lower','upper']]
                tag=f'f{factor:g}_w{width:g}_a{affected:g}';saved[tag+'_target']=t
                for h in [.0025,.01]:
                    prefix=tag+f'_h{h:g}';policies=controls(b,t,eta,h)
                    for name,p in policies.items():saved[prefix+'_p_'+name]=p
                    for placement,low,high,scale in placements(t,eta,l,u,base,True):
                        x=prefix+'_'+placement;saved[x+'_lower']=low;saved[x+'_upper']=high
                        for comp in ['Common','UC']:
                            p,rec=robust_common_binary(b,low,high,h,comparator=policies[comp],tau=TAU)
                            saved[x+'_p_R'+comp]=p
                        meta.append(dict(family='Finite',source=base,draw=draw,rarity=rarity,bias=factor,width=width,affected=affected,strength='',density='',radius='',gamma='',h=h,placement=placement,n=len(b),prefix=prefix,interval_prefix=x,scale=scale,width_target=float((u-l).mean()),file=base+'.npz'))
            np.savez_compressed(native/(base+'.npz'),**saved)
            print(json.dumps(dict(stage='finite_predictions',source=base,seconds=round(time.time()-start,1))),flush=True)
    for index,path in enumerate(sorted(TREE.glob('*.npz'))):
        src=np.load(path);use=~src['observed'];b=src['b'][use];eta=src['eta'][use];t=src['target'][use]
        parts=path.stem.split('_');draw=int(parts[0][1:]);strength=float(parts[1][1:]);density=float(parts[2][1:]);radius=int(parts[3][1:])
        saved={k:src[k] for k in ['fields','edges','couplings','observed','labels']};saved.update(b=b,eta=eta,target=t,global_vertices=np.flatnonzero(use))
        policies=controls(b,t,eta,.0025)
        for name,p in policies.items():saved['base_p_'+name]=p
        for multiplier in [0.,.25,1.]:
            l=src[f'L_g{multiplier:g}'];u=src[f'U_g{multiplier:g}']
            for placement,low,high,scale in placements(t,eta,l,u,path.stem,False):
                x=f'g{multiplier:g}_'+placement;saved[x+'_lower']=low;saved[x+'_upper']=high
                for comp in ['Common','UC']:
                    p,rec=robust_common_binary(b,low,high,.0025,comparator=policies[comp],tau=TAU)
                    saved[x+'_p_R'+comp]=p
                meta.append(dict(family='Tree',source=f'd{draw:02d}',draw=draw,rarity='',bias='',width='',affected='',strength=strength,density=density,radius=radius,gamma=multiplier,h=.0025,placement=placement,n=len(b),prefix='base',interval_prefix=x,scale=scale,width_target=float((u-l).mean()),file=path.name))
        np.savez_compressed(native/path.name,**saved)
        if index%27==26:print(json.dumps(dict(stage='tree_predictions',files=index+1,seconds=round(time.time()-start,1))),flush=True)
    write(ROOT/'prediction_metadata_v22.csv',meta)
    payloads={p.name:sha(p) for p in sorted(native.glob('*.npz'))}
    (ROOT/'prediction_commit_v22.json').write_text(json.dumps(dict(status='ALL_PROBABILITY_MAPS_COMMITTED_BEFORE_NEW_JOINT_RESPONSE_GENERATION',files=len(payloads),conditions=len(meta),payload_sha256=payloads,metadata_sha256=sha(ROOT/'prediction_metadata_v22.csv')),indent=2)+'\n')
    return meta

def sample_conditional_tree(data,rng):
    fields,edges,j,obs,labels=[data[k] for k in ['fields','edges','couplings','observed','labels']]
    eta,messages,neighbors=exact_tree(fields,edges,j,obs,labels)
    ys=np.zeros((JOINT,len(fields)),np.uint8)
    ys[:,0]=labels[0] if obs[0] else rng.random(JOINT)<eta[0]
    for v in range(1,len(fields)):
        if obs[v]:ys[:,v]=labels[v];continue
        parent=(v-1)//2
        jj=next(val for w,val in neighbors[v] if w==parent)
        score=fields[v]*np.array([-1.,1.])
        for w,val in neighbors[v]:
            if w!=parent:score+=messages[(w,v)]
        logits=(score[1]-score[0])+2*jj*(2*ys[:,parent].astype(float)-1)
        ys[:,v]=rng.random(JOINT)<1/(1+np.exp(-logits))
    assert np.all(ys[:,obs]==labels[obs])
    return ys[:,~obs]

def evaluate(meta):
    start=time.time();rows=[];bounds=[];joint_records={};tail_vectors={};metric_cache={};current=None
    rng=np.random.default_rng(2026102044)
    # Tree configurations that differ only in local radius share a conditional law.
    for m in meta:
        path=ROOT/'native'/m['file']
        if current!=path:
            data=np.load(path);current=path
        b,eta=data['b'],data['eta'];prefix=m['prefix'];x=m['interval_prefix']
        t=data[prefix.rsplit('_h',1)[0]+'_target'] if m['family']=='Finite' else data['target']
        law=m['file'].replace('_r0.npz','').replace('_r2.npz','').replace('_r10.npz','') if m['family']=='Tree' else m['source']
        if law not in joint_records:
            ys=(rng.random((JOINT,len(b)))<eta).astype(np.uint8) if m['family']=='Finite' else sample_conditional_tree(data,rng)
            joint_records[law]=ys
        ys=joint_records[law];low,high=data[x+'_lower'],data[x+'_upper']
        names=['Input','Native','Cap','Common','UC','Point','Oracle']+[f'PC{a:g}' for a in [.25,.5,.75]]+[f'PU{a:g}' for a in [.25,.5,.75]]
        policies={name:data[prefix+'_p_'+name] for name in names}
        policies.update({name:data[x+'_p_'+name] for name in ['RCommon','RUC']})
        true_input=float(ce(eta,b).mean());risk_common=float(ce(eta,policies['Common']).mean());risk_uc=float(ce(eta,policies['UC']).mean());risk_point=float(ce(eta,policies['Point']).mean())
        t1={}
        for name,p in policies.items():
            cachekey=(m['file'],prefix,name,x if name in ['RCommon','RUC'] else '')
            if cachekey not in metric_cache:
                with np.errstate(divide='ignore',invalid='ignore'):
                    delta1=np.log(b)-np.log(p);delta0=np.log1p(-b)-np.log1p(-p)
                    cost=np.maximum(delta1,delta0)
                    harm=ys*np.maximum(delta1,0.)+(1-ys)*np.maximum(delta0,0.)
                    expected_harm=eta*np.maximum(delta1,0.)+(1-eta)*np.maximum(delta0,0.)
                # Stable products for native zero/one endpoints.
                harm=np.where(ys==1,np.maximum(delta1,0.),np.maximum(delta0,0.))
                k=max(1,int(np.ceil(.01*len(b))));tv=np.partition(harm,len(b)-k,axis=1)[:,-k:].mean(axis=1)
                risk=float(ce(eta,p).mean())
                vals=dict(risk=risk,gain=true_input-risk,riskdiff_common=risk-risk_common,riskdiff_uc=risk-risk_uc,riskdiff_point=risk-risk_point,mean_cost=float(cost.mean()),max_cost=float(cost.max()),expected_harm=float(np.mean(expected_harm)),expected_T1=float(tv.mean()),T1_mcse=float(tv.std(ddof=1)/np.sqrt(JOINT)),equal_common=float(np.mean(np.abs(p-policies['Common'])<=1e-12)),equal_uc=float(np.mean(np.abs(p-policies['UC'])<=1e-12)),nonidentity=float(np.any(np.abs(p-b)>1e-12)),rare_cost=float(cost[b<.02].mean()) if np.any(b<.02) else 0.)
                if name not in ['Native','Cap']:
                    assert vals['mean_cost']<=m['h']+3e-13,(m,name,vals)
                    assert vals['max_cost']<=TAU+3e-13
                    assert risk>=float(ce(eta,policies['Oracle']).mean())-3e-12
                metric_cache[cachekey]=(vals,tv)
            vals,tv=metric_cache[cachekey];t1[name]=tv
            row={k:v for k,v in m.items() if k not in ['file','prefix','interval_prefix','scale']};row.update(method=name,**vals)
            rows.append(row)
        for name,comp in [('RCommon','Common'),('RUC','UC')]:
            p,q=policies[name],policies[comp];departure=logit(p)-logit(q)
            vr=float(np.mean(worst_excess_loss(p,q,low,high)))
            xi=float(np.mean(np.maximum(np.maximum(low-eta,eta-high),0)*np.abs(departure)))
            xid=float(np.mean(np.maximum(low-eta,0)*np.maximum(departure,0)+np.maximum(eta-high,0)*np.maximum(-departure,0)))
            riskdiff=float((ce(eta,p)-ce(eta,q)).mean())
            assert riskdiff<=vr+xid+2e-12,(m,name,riskdiff,vr,xid)
            assert xid<=xi+2e-12 and vr<=2e-12
            diff=t1[name]-t1[comp]
            out={k:v for k,v in m.items() if k not in ['file','prefix','interval_prefix']}
            out.update(method=name,comparator=comp,coverage=float(np.mean((eta>=low-3e-14)&(eta<=high+3e-14))),containment=float(np.all((eta>=low-3e-14)&(eta<=high+3e-14))),mean_width=float((high-low).mean()),width_residual=float((high-low).mean()-m['width_target']),width_q10=float(np.quantile(high-low,.1)),width_q50=float(np.quantile(high-low,.5)),width_q90=float(np.quantile(high-low,.9)),clipped_fraction=float(np.mean((low==0)|(high==1))),robust_value=vr,xi_u=xi,xi_dir=xid,bound=vr+xid,certified=float(vr+xid<=1e-12),nonidentity=float(np.max(np.abs(p-q))>1e-12),nonidentity_certificate=float(vr+xid<=1e-12 and np.max(np.abs(p-q))>1e-12),riskdiff=riskdiff,risk_worse=float(riskdiff>1e-12),expected_T1_difference=float(diff.mean()),paired_T1_mcse=float(diff.std(ddof=1)/np.sqrt(JOINT)),logit_departure=float(np.abs(departure).mean()))
            bounds.append(out)
        if len(rows)%10000<15:print(json.dumps(dict(stage='scoring',rows=len(rows),seconds=round(time.time()-start,1))),flush=True)
    (ROOT/'results').mkdir(exist_ok=True)
    write(ROOT/'results/method_metrics_v22.csv',rows);write(ROOT/'results/interval_diagnostics_v22.csv',bounds)
    np.savez_compressed(ROOT/'joint_response_draws_v22.npz',**joint_records)
    (ROOT/'run_receipt_v22.json').write_text(json.dumps(dict(status='PASS',complete=True,conditions=len(meta),method_rows=len(rows),bounds_rows=len(bounds),joint_laws=len(joint_records),joint_draws=JOINT,protocol_sha256=sha(ROOT/'protocol_v22.json'),response_sha256=sha(ROOT/'joint_response_draws_v22.npz'),seconds=time.time()-start),indent=2)+'\n')

def cli():
    import argparse
    global ROOT, FINITE, TREE, DRAWS, JOINT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finite-input',type=Path,required=True)
    parser.add_argument('--tree-input',type=Path,required=True,help='The native subdirectory written by the tree experiment.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=20)
    parser.add_argument('--joint-repetitions',type=int,default=512)
    args=parser.parse_args()
    ROOT=args.output.resolve();ROOT.mkdir(parents=True,exist_ok=True)
    FINITE=args.finite_input.resolve();TREE=args.tree_input.resolve()
    DRAWS=args.draws;JOINT=args.joint_repetitions
    if DRAWS<1 or JOINT<2:parser.error('Use positive draws and at least two joint repetitions.')
    if not list(TREE.glob('*.npz')):parser.error('The tree input directory contains no NPZ files.')
    seal();meta=prediction_phase();evaluate(meta)

if __name__=='__main__':cli()
