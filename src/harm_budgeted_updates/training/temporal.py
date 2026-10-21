"""Chronological local-tree forecasts with one-step delayed responses."""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from .data import load_graph, save_record, array_sha256

def probability(p):return np.clip(np.asarray(p,np.float64),1e-15,1-1e-15)

def model(kind,p):
 if kind=='HGBLocal':return HistGradientBoostingClassifier(**p['models'][kind])
 return RandomForestClassifier(**p['models'][kind])

def calibrate(raw,y,mask):
 z=logit(probability(raw));yy=y[mask]
 def obj(v):
  a=np.exp(-v[0]);h=a*z[mask]+v[1];q=expit(h);d=q-yy
  return float(np.mean(np.logaddexp(0,h)-yy*h)+.005*np.dot(v,v)),np.array([np.mean(d*(-a*z[mask])),np.mean(d)])+.01*v
 fit=minimize(obj,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-2,2),(-5,5)],options={'ftol':1e-12,'gtol':1e-9,'maxiter':200})
 assert np.isfinite(fit.x).all()
 return dict(parameters=fit.x.tolist(),success=bool(fit.success),objective=float(fit.fun),n=int(mask.sum()),optimizer_message=str(fit.message))

def applycal(raw,r):return probability(expit(np.exp(-r['parameters'][0])*logit(probability(raw))+r['parameters'][1]))


MODEL_PARAMETERS = {
    "HGBLocal": dict(max_iter=200, max_depth=3, learning_rate=.05, l2_regularization=1., random_state=1729, early_stopping=False),
    "RFLocal": dict(n_estimators=300, max_features="sqrt", min_samples_leaf=2, n_jobs=1, random_state=1729),
}


def run(data_path, output, origins=range(35, 50), references=("HGBLocal", "RFLocal"), feature_start=0, feature_stop=93):
    """Fit time-censored models and emit every current transaction for allocation.

    The response-arrival convention is a specified one-step delay. It does not
    claim that the public data contain actual adjudication timestamps.
    """
    data = load_graph(data_path)
    if "time_step" not in data:
        raise ValueError("Temporal input requires time_step with one integer per node.")
    steps = np.asarray(data["time_step"], dtype=int)
    if steps.shape != data["y"].shape:
        raise ValueError("time_step must have shape [n].")
    x = data["x"][:, feature_start:feature_stop]
    if x.shape[1] != 93:
        raise ValueError("The original temporal protocol uses exactly 93 local attributes.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    origins = sorted(set(origins))
    configuration = {"models": MODEL_PARAMETERS}
    for reference in references:
        fitted = {}
        records = {}
        for cutoff in sorted({value - offset for value in origins for offset in (1, 2)}):
            visible = data["y"].copy()
            visible[steps > cutoff] = -1
            train = (steps <= cutoff - 3) & (visible >= 0)
            calibration = (steps >= cutoff - 2) & (steps <= cutoff) & (visible >= 0)
            if not train.any() or not calibration.any() or len(np.unique(visible[train])) < 2:
                raise ValueError("A chronological fit lacks sufficient historical supervision.")
            fitted_model = model(reference, configuration)
            fitted_model.fit(x[train], visible[train])
            raw = probability(fitted_model.predict_proba(x[calibration])[:, list(fitted_model.classes_).index(1)])
            cal = calibrate(raw, visible[calibration], np.ones(calibration.sum(), dtype=bool))
            fitted[cutoff] = (fitted_model, cal)
            records[str(cutoff)] = dict(training_labels=int(train.sum()), calibration_labels=int(calibration.sum()), fit_max_time=int(steps[train].max()), calibration_min_time=int(steps[calibration].min()), calibration_max_time=int(steps[calibration].max()), calibration=cal, training_mask_sha256=array_sha256(train), calibration_mask_sha256=array_sha256(calibration), current_and_future_responses_erased=True)
        for origin in origins:
            current = steps == origin
            if not current.any():
                raise ValueError("An evaluation origin has no current transactions.")
            forecasts = []
            for cutoff in (origin - 2, origin - 1):
                fitted_model, cal = fitted[cutoff]
                raw = probability(fitted_model.predict_proba(x[current])[:, list(fitted_model.classes_).index(1)])
                forecasts.append(applycal(raw, cal))
            filename = "elliptic__step_" + str(origin) + "__" + reference + ".npz"
            np.savez_compressed(output / filename, b=forecasts[0], t=forecasts[1], y=data["y"][current], ids=data["ids"][current], graph=np.asarray("elliptic"), draw=np.asarray(origin), reference=np.asarray(reference), experiment=np.asarray("chronological"), time_step=steps[current])
        save_record(output / (reference + "__chronological_training.json"), dict(models=records, origins=origins, base_cutoff="origin minus 2", target_cutoff="origin minus 1", allocation_scope="All transactions in the current time step, including unknown responses.", evaluation_scope="Later known responses only.", feature_slice=[feature_start, feature_stop], historical_label_delay=1, actual_adjudication_timestamps_available=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--origins", nargs="+", type=int, default=list(range(35, 50)))
    parser.add_argument("--references", nargs="+", choices=list(MODEL_PARAMETERS), default=list(MODEL_PARAMETERS))
    parser.add_argument("--feature-start", type=int, default=0)
    parser.add_argument("--feature-stop", type=int, default=93)
    args = parser.parse_args()
    run(args.data, args.output, args.origins, args.references, args.feature_start, args.feature_stop)


if __name__ == "__main__":
    main()
