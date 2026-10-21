"""Exact binary allocation under a label-free D-infinity cost budget.

All optimization inputs are probabilities or nonnegative batch weights.  This
module accepts no evaluation labels.  The allowance constrains the supplied
batch itself, so it must be recomputed when the deployment batch changes.
"""
import numpy as np


VERSION = 'binary_global_dinfinity_v15_1'
TAU = .1
# Public mathematical operating points, not selected from empirical outcomes.
DENSE_ALLOWANCES = (0., .0001, .00025, .0005, .00075, .001, .0015, .002,
                    .0025, .003, .004, .005, .0065, .008, .01, .015,
                    .02, .03, .05, .075, .1)
BISECTIONS = 100


def _inputs(reference, target, weights=None):
    b = np.asarray(reference, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if b.ndim != 1 or not len(b) or t.shape != b.shape:
        raise ValueError('Matching nonempty one-dimensional probabilities required')
    if not np.isfinite(b).all() or np.any((b <= 0.) | (b >= 1.)):
        raise ValueError('Reference probabilities must be finite and strictly between zero and one')
    if not np.isfinite(t).all() or np.any((t < 0.) | (t > 1.)):
        raise ValueError('Target probabilities must be finite and between zero and one')
    if weights is None:
        w = np.full(len(b), 1. / len(b))
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != b.shape or not np.isfinite(w).all() or np.any(w < 0.):
            raise ValueError('Matching finite nonnegative batch weights required')
        scale = float(w.max())
        if scale <= 0.:
            raise ValueError('Batch weights must have positive total mass')
        w = w / scale
        w = w / float(w.sum())
    return b, t, w


def _tau_array(value, b):
    tau = np.broadcast_to(np.asarray(value, np.float64), b.shape)
    if not np.isfinite(tau).all() or np.any(tau < 0.):
        raise ValueError('Classwise loss caps must be finite and nonnegative')
    return tau


def dinfinity_binary(reference, prediction):
    """Per-node max_y log(b(y)/p(y)), computed without subtracting close logs."""
    b = np.asarray(reference, np.float64)
    p = np.asarray(prediction, np.float64)
    if p.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(p).all():
        raise ValueError('Matching finite probabilities required')
    if np.any((b <= 0.) | (b >= 1.) | (p <= 0.) | (p >= 1.)):
        raise ValueError('Open probabilities required')
    out = np.zeros_like(b)
    up, down = p > b, p < b
    close_up = up & ((p - b) <= .125 * (1. - b))
    close_down = down & ((b - p) <= .125 * b)
    out[close_up] = -np.log1p(-(p[close_up] - b[close_up]) / (1. - b[close_up]))
    out[close_down] = -np.log1p((p[close_down] - b[close_down]) / b[close_down])
    far_up, far_down = up & ~close_up, down & ~close_down
    out[far_up] = np.log1p(-b[far_up]) - np.log1p(-p[far_up])
    out[far_down] = np.log(b[far_down]) - np.log(p[far_down])
    return out


def floor_bounds(reference, tau=TAU, tau0=None, tau1=None):
    """Bounds from p(1)>=exp(-tau1)b(1), p(0)>=exp(-tau0)b(0).

    Scalar or per-node class caps are accepted.  Representable bounds are moved
    toward the reference only when necessary to retain numerical feasibility.
    """
    b = np.asarray(reference, np.float64)
    if b.ndim != 1 or not len(b) or not np.isfinite(b).all() or np.any((b <= 0.) | (b >= 1.)):
        raise ValueError('A nonempty open reference vector required')
    a0 = _tau_array(tau if tau0 is None else tau0, b)
    a1 = _tau_array(tau if tau1 is None else tau1, b)
    lower = np.maximum(b * np.exp(-a1), np.nextafter(0., 1.))
    upper = np.minimum(b + (-np.expm1(-a0)) * (1. - b), np.nextafter(1., 0.))
    lower = np.minimum(lower, b)
    upper = np.maximum(upper, b)
    for _ in range(16):
        bad1 = dinfinity_binary(b, lower) > a1
        bad0 = dinfinity_binary(b, upper) > a0
        if not bad1.any() and not bad0.any():
            return lower, upper
        lower[bad1] = np.nextafter(lower[bad1], b[bad1])
        upper[bad0] = np.nextafter(upper[bad0], b[bad0])
    raise ArithmeticError('Numerically feasible classwise floor could not be formed')


def binary_cross_entropy(target, prediction, weights=None):
    t = np.asarray(target, np.float64)
    p = np.asarray(prediction, np.float64)
    value = -(t * np.log(p) + (1. - t) * np.log1p(-p))
    return float(value.mean() if weights is None else np.asarray(weights) @ value)


def _at_multiplier(b, t, multiplier, lower, upper):
    if not np.isfinite(multiplier) or multiplier < 0.:
        raise ValueError('A finite nonnegative multiplier required')
    inverse = 1. / (1. + float(multiplier))
    left = t * inverse
    right = (t + float(multiplier)) * inverse
    unconstrained = np.minimum(np.maximum(b, left), right)
    return np.minimum(np.maximum(unconstrained, lower), upper)


def _allowance(value):
    if not np.isscalar(value) or not np.isfinite(value) or value < 0.:
        raise ValueError('A finite nonnegative batch allowance required')
    return float(value)


def _record(b, t, p, w, allowance, lower, upper, **values):
    cost = dinfinity_binary(b, p)
    return dict(version=VERSION, allowance=float(allowance),
                mean_cost=float(w @ cost), maximum_cost=float(cost.max()),
                target_cross_entropy=binary_cross_entropy(t, p, w),
                reference_target_cross_entropy=binary_cross_entropy(t, b, w),
                changed_fraction=float(np.mean(p != b)),
                floor_endpoint_mean_cost=float(w @ dinfinity_binary(b, np.clip(t, lower, upper))),
                batch_size=int(len(b)), weighted=not np.all(w == w[0]),
                **values)


def allocate_binary(reference, target, allowance, tau=TAU, weights=None,
                    tau0=None, tau1=None):
    """Minimize weighted CE(target,p) with mean Dinf<=allowance and node caps.

    Returns (full_batch_prediction, JSON-serializable optimization record).
    A returned dual bracket always uses its feasible (larger multiplier) end.
    This optimization uses the distribution target, never realized labels.
    """
    b, t, w = _inputs(reference, target, weights)
    h = _allowance(allowance)
    lower, upper = floor_bounds(b, tau, tau0, tau1)
    endpoint = np.clip(t, lower, upper)
    endpoint_cost = float(w @ dinfinity_binary(b, endpoint))
    if h == 0. or endpoint_cost == 0.:
        p = b.copy()
        return p, _record(b, t, p, w, h, lower, upper,
                          method='allocation', multiplier=None, location='identity',
                          dual_lower=None, dual_upper=None, iterations=0)
    if endpoint_cost <= h:
        return endpoint, _record(b, t, endpoint, w, h, lower, upper,
                                 method='allocation', multiplier=0., location='endpoint',
                                 dual_lower=0., dual_upper=0., iterations=0)
    lo, hi = 0., 1.
    bracket_steps = 0
    while float(w @ dinfinity_binary(b, _at_multiplier(b, t, hi, lower, upper))) > h:
        lo = hi
        hi *= 2.
        bracket_steps += 1
        if not np.isfinite(hi):
            raise ArithmeticError('Finite multiplier bracket unavailable at requested numerical scale')
    iterations = 0
    for _ in range(BISECTIONS):
        mid = lo + (hi - lo) / 2.
        if mid == lo or mid == hi:
            break
        iterations += 1
        p = _at_multiplier(b, t, mid, lower, upper)
        if float(w @ dinfinity_binary(b, p)) <= h:
            hi = mid
        else:
            lo = mid
    p = _at_multiplier(b, t, hi, lower, upper)
    if float(w @ dinfinity_binary(b, p)) > h:
        raise ArithmeticError('Feasible multiplier bracket lost')
    return p, _record(b, t, p, w, h, lower, upper,
                      method='allocation', multiplier=float(hi), location='interior',
                      dual_lower=float(lo), dual_upper=float(hi),
                      iterations=iterations, bracket_steps=bracket_steps)


def common_strength_binary(reference, target, allowance, tau=TAU, weights=None,
                           tau0=None, tau1=None):
    """Matched CE optimum on p=b+u*(floor(target)-b), 0<=u<=1.

    Each endpoint lies between b and target.  Consequently CE(target,p)
    decreases with u, and the exact scalar solution is the largest feasible u.
    The objective, batch, node caps, weights, and global cost match allocation.
    """
    b, t, w = _inputs(reference, target, weights)
    h = _allowance(allowance)
    lower, upper = floor_bounds(b, tau, tau0, tau1)
    endpoint = np.clip(t, lower, upper)
    direction = endpoint - b
    endpoint_cost = float(w @ dinfinity_binary(b, endpoint))
    if h == 0. or endpoint_cost == 0.:
        p, strength, location, iterations = b.copy(), 0., 'identity', 0
    elif endpoint_cost <= h:
        p, strength, location, iterations = endpoint, 1., 'endpoint', 0
    else:
        lo, hi, iterations = 0., 1., 0
        for _ in range(BISECTIONS):
            mid = lo + (hi - lo) / 2.
            if mid == lo or mid == hi:
                break
            iterations += 1
            candidate = b + mid * direction
            if float(w @ dinfinity_binary(b, candidate)) <= h:
                lo = mid
            else:
                hi = mid
        strength, location = lo, 'interior'
        p = b + strength * direction
    if float(w @ dinfinity_binary(b, p)) > h:
        raise ArithmeticError('Feasible common-strength bracket lost')
    return p, _record(b, t, p, w, h, lower, upper,
                      method='common_strength', strength=float(strength),
                      location=location, iterations=iterations)


def binary_information_projection(reference, lower, upper):
    """Argmin_{q in [lower,upper]} KL(Bern(q)||Bern(reference))."""
    b = np.asarray(reference, np.float64)
    lo = np.asarray(lower, np.float64)
    hi = np.asarray(upper, np.float64)
    if b.ndim != 1 or not len(b) or lo.shape != b.shape or hi.shape != b.shape:
        raise ValueError('Matching nonempty box and reference vectors required')
    if not np.isfinite(b).all() or np.any((b <= 0.) | (b >= 1.)):
        raise ValueError('Open reference probabilities required')
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any((lo < 0.) | (hi > 1.) | (lo > hi)):
        raise ValueError('Closed nonempty binary intervals required')
    return np.clip(b, lo, hi)


def box_worst_gain_binary(reference, prediction, lower, upper):
    """Per-node min_{pi in [lower,upper]} E_pi log(p(Y)/b(Y))."""
    b = np.asarray(reference, np.float64)
    p = np.asarray(prediction, np.float64)
    q = binary_information_projection(b, lower, upper)
    if p.shape != b.shape or not np.isfinite(p).all() or np.any((p <= 0.) | (p >= 1.)):
        raise ValueError('Matching open predictions required')
    # The adverse endpoint is selected by the actual prediction direction.
    adverse = np.where(p > b, np.asarray(lower), np.where(p < b, np.asarray(upper), q))
    log1 = np.log1p((p - b) / b)
    log0 = np.log1p(-(p - b) / (1. - b))
    return adverse * log1 + (1. - adverse) * log0


def allocate_box_binary(reference, lower, upper, allowance, tau=TAU,
                        weights=None, tau0=None, tau1=None):
    """Maximize the product-box worst gain using its common adverse projection.

    This supplies a model-game guarantee for the stated box, separate from
    label-free Dinf feasibility, which holds without any containment assumption.
    """
    q = binary_information_projection(reference, lower, upper)
    p, record = allocate_binary(reference, q, allowance, tau, weights, tau0, tau1)
    _, _, w = _inputs(reference, q, weights)
    gain = box_worst_gain_binary(reference, p, lower, upper)
    record.update(model_worst_gain=float(w @ gain),
                  minimum_node_model_gain=float(gain.min()),
                  box_identity_fraction=float(np.mean(q == np.asarray(reference))))
    return p, record
