"""Allocate first, then evaluate frozen probability pairs on known responses."""
from pathlib import Path
import argparse, json
import numpy as np
from .allocation import all_updates, dinfinity_binary
from .bayesian import confusion, beliefs
from .metrics import BinaryMetricContext
from .queues import evaluate_queue
from .io import array_hash, file_hash, scalar, write_csv, write_json


def input_files(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.npz"))
    if not files:
        raise ValueError("No NPZ cases found")
    return files


def allocate_case(source, destination, allowances, tau, kappa, cap_level):
    """Read probabilities and observed calibration roles; never read y."""
    with np.load(source, allow_pickle=False) as z:
        b = np.asarray(z["b"], float)
        t = np.asarray(z["t"], float)
        if (b.ndim != 1 or b.shape != t.shape or not len(b)
                or not np.all(np.isfinite(b)) or not np.all(np.isfinite(t))
                or np.any((b <= 0) | (b >= 1) | (t <= 0) | (t >= 1))):
            raise ValueError("Raw-target scoring requires finite open b and t")
        ids = np.asarray(z["ids"]) if "ids" in z.files else np.arange(len(b))
        if ids.shape != b.shape or len(np.unique(ids)) != len(ids):
            raise ValueError("Matching unique case identifiers required")
        order = np.argsort(ids, kind="stable")
        b, t = b[order], t[order]
        metadata = dict(experiment=str(scalar(z,"experiment","primary")),
                        graph=str(scalar(z,"graph",source.stem)),
                        draw=int(scalar(z,"draw",0)),
                        reference=str(scalar(z,"reference","predictor")))
        lo = z["lower"][order] if "lower" in z.files else None
        hi = z["upper"][order] if "upper" in z.files else None
        extra = {}
        if all(k in z.files for k in ["cal_old", "cal_new", "cal_y"]):
            prior, matrices = confusion(z["cal_y"], z["cal_old"], z["cal_new"])
            for mode, soft in [("hard", False), ("soft", True)]:
                _, extra["Bayes-"+mode] = beliefs(b,t,prior,matrices,soft)
        arrays = dict(b=b,t=t,ids=ids[order],source_order=order)
        rows = []
        for j,h in enumerate(allowances):
            outputs, recipes = all_updates(b,t,h,tau,kappa,lo,hi,cap_level)
            outputs.update(extra)
            for name,q in outputs.items():
                key = "h"+str(j)+"__"+name
                arrays[key] = q
                cost = dinfinity_binary(b,q)
                rows.append(dict(key=key,method=name,h=h,tau=tau,
                                 prediction_sha256=array_hash(q),
                                 C_deploy=float(cost.mean()),
                                 maximum_cost_deploy=float(cost.max()),
                                 recipe=recipes.get(name,{})))
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination/"predictions.npz", **arrays)
    record = dict(source_name=source.name,source_sha256=file_hash(source),
                  metadata=metadata,conditions=rows,
                  evaluation_responses_opened=False)
    write_json(destination/"allocation.json",record)
    return record


def evaluate_case(source, destination):
    record = json.loads((destination/"allocation.json").read_text())
    if file_hash(source) != record["source_sha256"]:
        raise ValueError("Input changed after allocation")
    with np.load(source,allow_pickle=False) as src, np.load(
            destination/"predictions.npz",allow_pickle=False) as z:
        y = np.asarray(src["y"],int)[z["source_order"]]
        if y.shape != z["b"].shape or not np.isin(y,[-1,0,1]).all():
            raise ValueError("Use y=-1 for unknown and y=0/1 for known responses")
        known = y >= 0
        if len(np.unique(y[known])) != 2:
            raise ValueError("AP/AUROC evaluation requires both known classes")
        context = BinaryMetricContext(z["b"][known], y[known])
        rows = []
        for condition in record["conditions"]:
            q = z[condition["key"]]
            if array_hash(q) != condition["prediction_sha256"]:
                raise ValueError("Prediction changed after allocation")
            metrics = context(q[known])
            cost = dinfinity_binary(z["b"],q)
            queue,_ = evaluate_queue(z["b"],q,y,.01,ids=z["ids"])
            rows.append(dict(**record["metadata"],method=condition["method"],
                             h=condition["h"],tau=condition["tau"],
                             C_deploy=float(cost.mean()),
                             C_known=float(cost[known].mean()),
                             maximum_cost_deploy=float(cost.max()),
                             n_deployment=len(y),**metrics,
                             **{"deploy_queue_"+k:v for k,v in queue.items()}))
    return rows


def main():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config",type=Path)
    config_arg,_ = preliminary.parse_known_args()
    defaults = json.loads(config_arg.config.read_text()) if config_arg.config else {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--h",type=float,nargs="+",default=defaults.get("h",[.0025]))
    p.add_argument("--tau",type=float,default=defaults.get("tau",.1))
    p.add_argument("--kappa",type=float,default=defaults.get("kappa",.5))
    p.add_argument("--cap-level",type=float,default=defaults.get("cap_level",.025))
    p.add_argument("--stage",choices=["allocate","evaluate","all"],default="all")
    args = p.parse_args()
    files = input_files(args.input)
    if len({f.stem for f in files}) != len(files):
        raise ValueError("Case filenames must be unique")
    if args.stage in ["allocate","all"]:
        args.output.mkdir(parents=True,exist_ok=True)
        records = [allocate_case(f,args.output/f.stem,args.h,args.tau,
                                 args.kappa,args.cap_level) for f in files]
        write_json(args.output/"allocation_complete.json",
                   dict(complete=True,cases=len(records),
                        evaluation_responses_opened=False))
    if args.stage in ["evaluate","all"]:
        done = json.loads((args.output/"allocation_complete.json").read_text())
        if not done["complete"] or done["cases"] != len(files):
            raise ValueError("Allocate all cases before evaluation")
        rows = []
        for source in files:
            rows.extend(evaluate_case(source,args.output/source.stem))
        write_csv(args.output/"metrics.csv",rows)
        print("Evaluated",len(rows),"method/budget/case conditions")


if __name__ == "__main__":
    main()
