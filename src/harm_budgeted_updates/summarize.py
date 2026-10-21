"""Paired graph/draw summaries that retain all budget points in each resample."""
from pathlib import Path
from collections import defaultdict
import argparse,csv
import numpy as np
from .io import write_csv,write_json

METRICS = ("nll","gain","AP","AUROC","recall_at_capacity","H","tail1",
           "C_deploy","C_known","negative_flip_0.1","positive_flip_0.1",
           "deploy_queue_known_lost_TP_per_capacity",
           "deploy_queue_known_enter_TP_per_capacity",
           "deploy_queue_all_lost_per_capacity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resamples",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=2027)
    parser.add_argument("--graphs",nargs="+")
    parser.add_argument("--temporal-graphs",nargs="*",default=["elliptic"])
    parser.add_argument("--block-length",type=int,default=3)
    parser.add_argument("--left",default="UC")
    parser.add_argument("--right",default="Logit")
    args = parser.parse_args()
    if args.resamples < 1 or args.block_length < 1:
        parser.error("Resample count and block length must be positive")
    with args.input.open(newline="",encoding="utf8") as f:
        records = list(csv.DictReader(f))
    if args.graphs:
        records = [r for r in records if r["graph"] in args.graphs]
    if not records:
        raise ValueError("No matching metric rows")
    fields = [m for m in METRICS if all(r.get(m,"")!="" for r in records)]
    grouped = defaultdict(list)
    for r in records:
        key=(r["experiment"],r["graph"],int(r["draw"]),float(r["h"]),r["method"])
        grouped[key].append([float(r[m]) for m in fields])
    # Predictor averaging is performed inside each acquisition before resampling.
    values={k:np.mean(v,axis=0) for k,v in grouped.items()}
    rng=np.random.default_rng(args.seed)
    indices={}; units={}
    for experiment,graph,*_ in sorted(values):
        key=(experiment,graph)
        if key in indices:
            continue
        draws=sorted({k[2] for k in values if k[:2]==key})
        n=len(draws);units[key]=draws
        if graph in args.temporal_graphs:
            starts=rng.integers(n,size=(args.resamples,int(np.ceil(n/args.block_length))))
            idx=((starts[:,:,None]+np.arange(args.block_length))%n).reshape(args.resamples,-1)[:,:n]
        else:
            idx=rng.integers(n,size=(args.resamples,n))
        indices[key]=idx
    summaries=[];macro_distributions={}
    for experiment in sorted({k[0] for k in values}):
        graphs=sorted({k[1] for k in values if k[0]==experiment})
        for h,method in sorted({(k[3],k[4]) for k in values if k[0]==experiment}):
            means=[];boot=[]
            for graph in graphs:
                keys=[(experiment,graph,d,h,method) for d in units[experiment,graph]]
                if not all(k in values for k in keys):
                    continue
                a=np.asarray([values[k] for k in keys]);bs=a[indices[experiment,graph]].mean(1)
                row=dict(experiment=experiment,scope=graph,h=h,method=method,acquisitions=len(a))
                for j,metric in enumerate(fields):
                    row[metric]=float(a[:,j].mean())
                    row[metric+"_lo"],row[metric+"_hi"]=np.quantile(bs[:,j],[.025,.975])
                summaries.append(row);means.append(a.mean(0));boot.append(bs)
            # A macro row requires every graph, so missing baselines cannot change scope.
            if len(means)==len(graphs):
                mean=np.mean(means,axis=0);bs=np.mean(boot,axis=0)
                row=dict(experiment=experiment,scope="macro",h=h,method=method,graphs=len(graphs))
                for j,metric in enumerate(fields):
                    row[metric]=float(mean[j]);row[metric+"_lo"],row[metric+"_hi"]=np.quantile(bs[:,j],[.025,.975])
                summaries.append(row);macro_distributions[experiment,h,method]=(mean,bs)
    contrasts=[]
    for experiment,h,method in sorted(macro_distributions):
        if method!=args.left or (experiment,h,args.right) not in macro_distributions:
            continue
        mean,bs=macro_distributions[experiment,h,method]
        other,otherbs=macro_distributions[experiment,h,args.right]
        row=dict(experiment=experiment,h=h,left=args.left,right=args.right)
        for j,metric in enumerate(fields):
            row[metric]=float(mean[j]-other[j])
            row[metric+"_lo"],row[metric+"_hi"]=np.quantile(bs[:,j]-otherbs[:,j],[.025,.975])
        contrasts.append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/"means.csv",summaries)
    write_csv(args.output/"paired_contrasts.csv",contrasts)
    np.savez_compressed(args.output/"bootstrap_indices.npz",
                        **{str(i):v for i,v in enumerate(indices.values())})
    write_json(args.output/"resampling.json",dict(seed=args.seed,resamples=args.resamples,
               temporal_graphs=args.temporal_graphs,block_length=args.block_length,
               unit_keys=[list(k) for k in indices],
               all_budgets_methods_and_predictors_retained_in_each_resampled_acquisition=True))
    print("Wrote",len(summaries),"summary rows and",len(contrasts),"paired contrasts")


if __name__=="__main__":
    main()
