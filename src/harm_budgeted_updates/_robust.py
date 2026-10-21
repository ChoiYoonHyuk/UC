"""Robust binary allocation relative to a feasible common-strength comparator.

The original reference supplies every D-infinity cost and nodewise cap.
The comparator supplies only the improvement objective.  All inputs are
probabilities, uncertainty intervals, or nonnegative batch weights; this
module neither accepts nor derives realized evaluation labels.
"""
import numpy as np
from ._binary import (
    _inputs, _allowance, _at_multiplier, floor_bounds, dinfinity_binary,
    common_strength_binary, allocate_binary, BISECTIONS,
)

VERSION = 'robust_common_baseline_v17_1'


def worst_excess_loss(prediction, comparator, lower, upper):
    """Nodewise max_theta CE(theta,p)-CE(theta,comparator)."""
    p, q = np.asarray(prediction, float), np.asarray(comparator, float)
    low, high = np.asarray(lower, float), np.asarray(upper, float)
    adverse = np.where(p >= q, low, high)
    log1 = np.log1p((p - q) / q)
    log0 = np.log1p(-(p - q) / (1. - q))
    return -(adverse * log1 + (1. - adverse) * log0)


def _intervals(lower, upper, b):
    low, high = np.asarray(lower, float), np.asarray(upper, float)
    if low.shape != b.shape or high.shape != b.shape:
        raise ValueError('Uncertainty intervals must match the reference')
    if (not np.isfinite(low).all() or not np.isfinite(high).all()
            or np.any(low < 0.) or np.any(high > 1.) or np.any(low > high)):
        raise ValueError('Nonempty probability intervals required')
    return low, high


def _dual_minimizer(b, q, low, high, multiplier, cap_low, cap_high):
    """Minimize the separable convex CE-offset plus multiplier times cost."""
    minus = np.clip(_at_multiplier(b, high, multiplier, cap_low, cap_high), cap_low, q)
    plus = np.clip(_at_multiplier(b, low, multiplier, cap_low, cap_high), q, cap_high)
    minus_value = worst_excess_loss(minus, q, low, high) + multiplier * dinfinity_binary(b, minus)
    plus_value = worst_excess_loss(plus, q, low, high) + multiplier * dinfinity_binary(b, plus)
    # The functions are strictly convex on each nonempty branch.  At an exact
    # branch tie both minimizers coincide at q.  Return q explicitly in that case.
    choose_minus, choose_plus = minus_value < plus_value, plus_value < minus_value
    return np.where(choose_minus, minus, np.where(choose_plus, plus, q))


def robust_common_binary(reference, lower, upper, allowance, comparator=None,
                         target=None, tau=.1, weights=None):
    """Minimize worst excess CE relative to comparator under original budget.

    Comparator must satisfy the same mean budget and node caps.  Omitting it
    constructs the ordinary probability-space common-strength update toward
    target.  A singleton box is dispatched to the original exact allocator;
    a full box returns the comparator byte-for-byte.  These are mathematical
    limiting cases, not data-dependent selection rules.
    """
    if comparator is None:
        if target is None:
            raise ValueError('Provide a feasible comparator or a target')
        q, _ = common_strength_binary(reference, target, allowance, tau, weights)
    else:
        q = np.asarray(comparator, np.float64)
    b, q, w = _inputs(reference, q, weights)
    if np.any((q <= 0.) | (q >= 1.)):
        raise ValueError('Open comparator probabilities required')
    low, high = _intervals(lower, upper, b)
    h = _allowance(allowance)
    cap_low, cap_high = floor_bounds(b, tau)
    comparator_cost = float(w @ dinfinity_binary(b, q))
    if np.any(q < cap_low) or np.any(q > cap_high) or comparator_cost > h + 8e-15:
        raise ValueError('Comparator must obey original reference caps and mean budget')
    multiplier, iterations, bracket_steps = None, 0, 0
    dual_lower, dual_upper = None, None
    if h == 0. or np.all(cap_low == cap_high):
        p, location = b.copy(), 'identity'
    elif np.all((low == 0.) & (high == 1.)):
        p, location = q.copy(), 'full_uncertainty'
    elif np.all(low == high):
        p, original = allocate_binary(b, low, h, tau, w)
        multiplier = original.get('multiplier')
        iterations = original.get('iterations', 0)
        dual_lower, dual_upper = original.get('dual_lower'), original.get('dual_upper')
        location = 'singleton_' + original['location']
    else:
        p = _dual_minimizer(b, q, low, high, 0., cap_low, cap_high)
        if float(w @ dinfinity_binary(b, p)) <= h:
            multiplier, location = 0., 'robust_endpoint'
        else:
            lo, hi = 0., 1.
            while float(w @ dinfinity_binary(b, _dual_minimizer(b, q, low, high, hi, cap_low, cap_high))) > h:
                lo, hi = hi, hi * 2.
                bracket_steps += 1
                if not np.isfinite(hi):
                    raise ArithmeticError('Finite robust multiplier bracket unavailable')
            for _ in range(BISECTIONS):
                mid = lo + (hi - lo) / 2.
                if mid == lo or mid == hi:
                    break
                iterations += 1
                candidate = _dual_minimizer(b, q, low, high, mid, cap_low, cap_high)
                if float(w @ dinfinity_binary(b, candidate)) <= h:
                    hi = mid
                else:
                    lo = mid
            p = _dual_minimizer(b, q, low, high, hi, cap_low, cap_high)
            multiplier, dual_lower, dual_upper = float(hi), float(lo), float(hi)
            location = 'interior'
    cost = dinfinity_binary(b, p)
    worst = float(w @ worst_excess_loss(p, q, low, high))
    # The comparator is always feasible and has objective zero.  Roundoff in
    # a dual root may produce a minute positive objective.  The exact feasible
    # comparator supplies a deterministic safeguard against numerical harm.
    numerical_fallback = False
    if worst > 0.:
        p, cost, worst = q.copy(), dinfinity_binary(b, q), 0.
        numerical_fallback = True
    if float(w @ cost) > h + 8e-15:
        raise ArithmeticError('Robust mean budget feasibility lost')
    return p, dict(version=VERSION, method='robust_common', location=location,
        allowance=h, mean_cost=float(w @ cost), maximum_cost=float(cost.max()),
        comparator_mean_cost=comparator_cost, worst_excess_loss=worst,
        certified_gain_over_comparator=-worst, multiplier=multiplier,
        dual_lower=dual_lower, dual_upper=dual_upper, iterations=iterations,
        bracket_steps=bracket_steps, numerical_fallback=numerical_fallback,
        changed_fraction=float(np.mean(p != b)),
        comparator_changed_fraction=float(np.mean(p != q)), batch_size=int(len(b)))


def logit_common_binary(reference, target, allowance, tau=.1, weights=None):
    """Largest feasible scalar step on the logit segment to capped target."""
    b, t, w = _inputs(reference, target, weights)
    h = _allowance(allowance)
    low, high = floor_bounds(b, tau)
    endpoint = np.clip(t, low, high)
    z = np.log(b) - np.log1p(-b)
    direction = np.log(endpoint) - np.log1p(-endpoint) - z
    def at(u):
        zz = z + u * direction
        pp = np.empty_like(zz)
        positive = zz >= 0.
        pp[positive] = 1. / (1. + np.exp(-zz[positive]))
        expz = np.exp(zz[~positive])
        pp[~positive] = expz / (1. + expz)
        return np.clip(pp, np.minimum(b, endpoint), np.maximum(b, endpoint))
    if h == 0. or not np.any(direction):
        p, strength = b.copy(), 0.
    elif float(w @ dinfinity_binary(b, endpoint)) <= h:
        p, strength = endpoint, 1.
    else:
        left, right = 0., 1.
        for _ in range(BISECTIONS):
            mid = left + (right - left) / 2.
            if mid == left or mid == right:
                break
            if float(w @ dinfinity_binary(b, at(mid))) <= h:
                left = mid
            else:
                right = mid
        p, strength = at(left), left
    return p, dict(method='logit_common', strength=float(strength),
                   mean_cost=float(w @ dinfinity_binary(b, p)), allowance=h)


def reliability_common_binary(reference, target, reliability, allowance, tau=.1,
                              weights=None):
    """Prespecified reliability-weighted segment with one feasible scalar."""
    b, t, w = _inputs(reference, target, weights)
    r = np.asarray(reliability, float)
    if r.shape != b.shape or not np.isfinite(r).all() or np.any((r < 0.) | (r > 1.)):
        raise ValueError('Reliability weights must be in [0,1]')
    low, high = floor_bounds(b, tau)
    endpoint = b + r * (np.clip(t, low, high) - b)
    # Passing the already capped endpoint preserves its intended weighted path.
    p, record = common_strength_binary(b, endpoint, allowance, tau, w)
    record['method'] = 'reliability_common'
    return p, record
