"""Tree predictors and the original affine-calibration, LP, and C&S pipeline."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from .data import load_graph, roles_from_data, save_record, array_sha256

def probability(p):
    return np.clip(p, 1e-15, 1 - 1e-15)

def loss(p, y):
    p = probability(p)
    return float(np.mean(-np.where(y == 1, np.log(p), np.log1p(-p))))

def affine_calibration(p,y,A):
    z=logit(probability(p));yy=y[A]
    def objective(v):
        a=np.exp(-v[0]);h=a*z[A]+v[1];q=expit(h);d=q-yy
        val=np.mean(np.logaddexp(0,h)-yy*h)+.005*np.dot(v,v)
        grad=np.array([np.mean(d*(-a*z[A])),np.mean(d)])+.01*v
        return val,grad
    fit=minimize(objective,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-2,2),(-5,5)],options={'ftol':1e-12,'gtol':1e-9,'maxiter':200})
    value=probability(expit(np.exp(-fit.x[0])*z+fit.x[1]))
    return value,dict(parameters=fit.x.tolist(),success=bool(fit.success),objective=float(fit.fun))

def propagation_inputs(predictions,adjacency,mean,masks,y):
    O=masks['train'];A=masks['calibrate'];n=len(O)
    calibrated={};calrecords={}
    for name,p in predictions.items():calibrated[name],calrecords[name]=affine_calibration(p,y,A)
    ensemble=probability(expit(np.mean(np.stack([logit(p)for p in calibrated.values()]),axis=0)))
    stack,calrecords['ensemble']=affine_calibration(ensemble,y,A)
    known=np.zeros((n,2));known[O]=np.eye(2)[y[O]]
    lp=known.copy()
    for _ in range(50):lp=.9*(mean@lp)+.1*known;lp[O]=known[O]
    total=lp.sum(1)
    lp=probability(np.divide(lp[:,1],total,out=np.full(n,.5),where=total>0))
    def choose(b,t):
        values=[0,.25,.5,.75,1];scores=[loss((1-a)*b[A]+a*t[A],y[A])for a in values]
        idx=min(range(len(values)),key=lambda i:(scores[i],values[i]))
        return probability((1-values[idx])*b+values[idx]*t),dict(alpha=values[idx],A_scores=scores)
    lpref,lprecord=choose(stack,lp)
    # Same fixed D^{-1}A autoscale C&S equations as the existing pipeline.
    probs=np.column_stack((1-lpref,lpref));residual=np.zeros_like(probs);residual[O]=np.eye(2)[y[O]]-probs[O]
    propagated=residual.copy()
    for _ in range(50):propagated=np.clip(.95*(mean@propagated)+.05*residual,-1,1)
    average=np.abs(residual[O]).sum()/O.sum();denominator=np.abs(propagated).sum(1)
    factor=np.divide(average,denominator,out=np.ones(n),where=denominator>0);factor[~np.isfinite(factor)|(factor>1000)]=1
    initial=probs+factor[:,None]*propagated;initial[O]=np.eye(2)[y[O]]
    smooth=initial.copy()
    for _ in range(50):smooth=np.clip(.95*(mean@smooth)+.05*initial,0,1)
    smooth=np.maximum(smooth,1e-15);smooth/=smooth.sum(1,keepdims=True)
    cs,csrecord=choose(lpref,probability(smooth[:,1]))
    return calibrated,dict(stack=stack,LP=lpref,CS=cs),dict(calibration=calrecords,LP=lprecord,CS=csrecord)

def seed(dataset,draw):return int(hashlib.sha256(('strong-fraud-v16|'+dataset+'|'+str(draw)).encode()).hexdigest()[:16],16)

def model(kind,s):
 if kind=='HGBGraph':return HistGradientBoostingClassifier(max_iter=200,max_depth=3,learning_rate=.05,l2_regularization=1.,random_state=s%(2**31),early_stopping=False)
 return RandomForestClassifier(n_estimators=300,max_features='sqrt',min_samples_leaf=2,n_jobs=1,random_state=s%(2**31))

def predict(m,X):
 p=m.predict_proba(X);ids=np.flatnonzero(m.classes_==1)
 return probability(p[:,ids[0]]if len(ids)else np.zeros(len(X)))


def enlarged_roles(masks, dataset, draw, target_count):
    """Original nested acquisition for the 500-to-5,000-label experiment."""
    fit = masks["train"] | masks["select"]
    evaluate = masks["test"].copy()
    if target_count < fit.sum():
        raise ValueError("target_count cannot remove already acquired labels.")
    amount = target_count - int(fit.sum())
    if amount:
        if amount >= evaluate.sum():
            raise ValueError("The enlarged acquisition leaves no evaluation nodes.")
        value = int(hashlib.sha256(("large-label-v26|" + dataset + "|" + str(draw)).encode()).hexdigest()[:16], 16)
        extra = np.random.default_rng(value).permutation(np.flatnonzero(evaluate))[:amount]
        fit[extra] = True
        evaluate[extra] = False
    return fit, evaluate


def run(data_path, output, dataset, draw, target_count=550, references=("HGBGraph", "RFGraph")):
    """Fit old/new predictors with disjoint training and calibration responses."""
    data = load_graph(data_path)
    masks = roles_from_data(data, dataset, draw)
    fit, evaluate = enlarged_roles(masks, dataset, draw, target_count)
    train = masks["train"]
    cal = masks["calibrate"]
    visible_y = np.where(train | fit | cal, data["y"], -1)
    if np.any(visible_y[evaluate] != -1):
        raise AssertionError("Evaluation responses entered the fitting view.")
    adjacency = data["adjacency"].astype(np.float64)
    degree = np.asarray(adjacency.sum(1)).ravel()
    mean = sparse.diags(1 / np.maximum(degree, 1)) @ adjacency
    design = np.column_stack((data["x"], mean @ data["x"])).astype(np.float32)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for reference in references:
        old = model(reference, seed(dataset, draw))
        old.fit(design[train], visible_y[train])
        old_raw = predict(old, design)
        _, old_inputs, old_stages = propagation_inputs({reference: old_raw}, adjacency, mean, masks, visible_y)
        updated = model(reference, seed(dataset, draw))
        updated.fit(design[fit], visible_y[fit])
        new_raw = predict(updated, design)
        new_masks = dict(masks, train=fit, select=np.zeros(len(train), dtype=bool), test=evaluate)
        _, new_inputs, new_stages = propagation_inputs({reference: new_raw}, adjacency, mean, new_masks, visible_y)
        b, t = old_inputs["CS"], new_inputs["CS"]
        filename = dataset + "__draw_" + str(draw) + "__" + reference + ".npz"
        experiment = "large_labels" if target_count > 550 else "primary"
        np.savez_compressed(output / filename, b=b[evaluate], t=t[evaluate], y=data["y"][evaluate], ids=data["ids"][evaluate], graph=np.asarray(dataset), draw=np.asarray(draw), reference=np.asarray(reference), experiment=np.asarray(experiment), cal_old=b[cal], cal_new=t[cal], cal_y=visible_y[cal])
        records.append(dict(reference=reference, training_old=int(train.sum()), training_new=int(fit.sum()), calibration_count=int(cal.sum()), evaluation_count=int(evaluate.sum()), old_stages=old_stages, new_stages=new_stages, b_sha256=array_sha256(b[evaluate]), t_sha256=array_sha256(t[evaluate]), evaluation_responses_used_for_fitting=False))
    save_record(output / (dataset + "__draw_" + str(draw) + "__training.json"), dict(dataset=dataset, draw=draw, role_hashes={key: array_sha256(value) for key, value in masks.items()}, records=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--target-count", type=int, default=550)
    parser.add_argument("--references", nargs="+", choices=["HGBGraph", "RFGraph"], default=["HGBGraph", "RFGraph"])
    args = parser.parse_args()
    run(args.data, args.output, args.dataset, args.draw, args.target_count, args.references)


if __name__ == "__main__":
    main()
