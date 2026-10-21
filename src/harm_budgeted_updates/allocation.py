"""Public allocation API; prediction maps use no evaluation responses."""
import numpy as np
from scipy.special import expit, logit
from ._binary import (
    allocate_binary, common_strength_binary, dinfinity_binary, floor_bounds,
    binary_cross_entropy, box_worst_gain_binary,
)
from ._paths import uniform_cap_binary, native_forward_kl_binary
from ._robust import robust_common_binary, worst_excess_loss
from ._point_backend import allocate_fast


def _logit_path(b, t, tau, alpha):
    """The exact probability map used for the added logit control."""
    v = np.clip(t, *floor_bounds(b, tau))
    if alpha == 0:
        return b.copy()
    if alpha == 1:
        return v
    return expit(logit(b) + alpha * (logit(v) - logit(b)))


def logit_matched_binary(b, t, h, tau=.1):
    """Match Common's cost on the same capped endpoint in logit space.

    This retains the 58-step search from the final baseline experiment.
    The coefficient depends only on the two probability vectors and budget.
    """
    b, t = np.asarray(b, float), np.asarray(t, float)
    v = np.clip(t, *floor_bounds(b, tau))
    capacity = float(dinfinity_binary(b, v).mean())
    goal = min(h, capacity)
    if capacity <= h:
        return v, dict(alpha=1., cost_target=goal)
    lo, hi = 0., 1.
    lb, step = logit(b), logit(v) - logit(b)
    for _ in range(58):
        mid = (lo + hi) / 2
        q = expit(lb + mid * step)
        if float(dinfinity_binary(b, q).mean()) <= goal:
            lo = mid
        else:
            hi = mid
    q = _logit_path(b, t, tau, lo)
    return q, dict(alpha=lo, cost_target=goal)


def cap_path_binary(b, t, h, tau=.1, level=.025, backend="cpu"):
    """Target-optimal update at max(UC level, requested individual cap)."""
    _, uc_record = uniform_cap_binary(b, t, h, tau)
    gamma_uc = uc_record["parameter"]
    gamma = min(tau, max(gamma_uc, level))
    if backend == "cpu":
        p, record = allocate_binary(b, t, h, tau=gamma)
    elif backend == "auto":
        p, record = allocate_fast(b, t, h, tau=gamma)
    else:
        raise ValueError("Backend must be cpu or auto")
    return p, dict(record, requested_cap=level, effective_gamma=gamma,
                   gamma_uc=gamma_uc)


def all_updates(b, t, h=.0025, tau=.1, kappa=.5, lower=None, upper=None,
                cap_level=.025, mixture_weights=(.25, .5, .75)):
    """Return paper methods without inspecting evaluation responses.

    Native, Average50 and CapOnly do not obey the aggregate budget. All
    remaining methods obey both the batch and individual constraints.
    If intervals are omitted, the real-graph sensitivity intervals are used.
    """
    b, t = np.asarray(b, float), np.asarray(t, float)
    if b.ndim != 1 or b.shape != t.shape or not len(b):
        raise ValueError("Matching nonempty probability vectors required")
    if not np.isfinite(h) or h < 0 or not np.isfinite(tau) or tau < 0:
        raise ValueError("Finite nonnegative budgets required")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("Finite nonnegative interval factor required")
    predictions = {"Input": b.copy(), "Native": t.copy(),
                   "Average50": (b + t) / 2,
                   "CapOnly": np.clip(t, *floor_bounds(b, tau))}
    records = {name: {} for name in predictions}
    for name, solver in [("Common", common_strength_binary),
                         ("Logit", logit_matched_binary),
                         ("UC", uniform_cap_binary),
                         ("Point", allocate_binary)]:
        predictions[name], records[name] = solver(b, t, h, tau=tau)
    predictions["CapPath"], records["CapPath"] = cap_path_binary(
        b, t, h, tau=tau, level=cap_level)
    if (lower is None) != (upper is None):
        raise ValueError("Provide both interval endpoints")
    if lower is None:
        radius = kappa * np.minimum(b, 1-b)
        lower, upper = np.maximum(0, t-radius), np.minimum(1, t+radius)
    for name, comparator in [("RCommon", "Common"), ("RUC", "UC")]:
        predictions[name], records[name] = robust_common_binary(
            b, lower, upper, h, comparator=predictions[comparator], tau=tau)
    for alpha in mixture_weights:
        if not 0 <= alpha <= 1:
            raise ValueError("Mixture weights must lie in [0,1]")
        for comparator in ["Common", "UC"]:
            name = "Point-" + comparator + "-" + str(alpha)
            predictions[name] = ((1-alpha)*predictions[comparator]
                                 + alpha*predictions["Point"])
            records[name] = dict(alpha=alpha, comparator=comparator)
    for name, p in predictions.items():
        if name in ["Native", "Average50", "CapOnly"]:
            continue
        c = dinfinity_binary(b, p)
        if c.mean() > h+2e-12 or c.max() > tau+2e-12:
            raise ArithmeticError("Budget violation for " + name)
    return predictions, records
