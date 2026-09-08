# -*- coding: utf-8 -*-
"""
Controls.

These are not competitors, they are the calibration of the comparison. A
metaheuristic that does not beat random sampling at the same budget is not
doing anything, and on a smooth one-dimensional box a dense grid is often the
honest answer. Reporting these alongside the metaheuristics is what keeps the
comparison in EXP-04 from being decorative.
"""
from __future__ import annotations

import numpy as np

from .base import INF, clamp, make_info, out_of_budget, safe_eval


def optimize_random(obj, lo, hi, budget, rng, x0=None, **_):
    """Uniform sampling in the box, same budget as everyone else."""
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    theta_star, J_star = None, INF
    while not out_of_budget(obj):
        cand = rng.uniform(lo, hi)
        J = safe_eval(obj, cand)
        if J == INF:
            break
        if J < J_star:
            theta_star, J_star = cand.copy(), J
    if theta_star is None:
        theta_star = 0.5 * (lo + hi)
        J_star = obj.best_so_far()
    return theta_star, make_info(obj, theta_star, J_star,
                                 dict(optimizer="random"))


def optimize_grid(obj, lo, hi, budget, rng, x0=None, log_spaced=True, **_):
    """Dense sweep of the box.

    Only sensible for one free radius; with K free radii a product grid needs
    ``m^K`` points, so the function falls back to a diagonal sweep and says so
    in ``info``, which is itself part of the argument for a search.
    """
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    K = lo.size
    n_points = int(budget) if budget else 50

    if log_spaced:
        grid = np.geomspace(lo[0], hi[0], n_points)
    else:
        grid = np.linspace(lo[0], hi[0], n_points)

    theta_star, J_star = None, INF
    for value in grid:
        if out_of_budget(obj):
            break
        cand = clamp(np.full(K, value), lo, hi)
        J = safe_eval(obj, cand)
        if J == INF:
            break
        if J < J_star:
            theta_star, J_star = cand.copy(), J
    if theta_star is None:
        theta_star = 0.5 * (lo + hi)
        J_star = obj.best_so_far()
    return theta_star, make_info(obj, theta_star, J_star,
                                 dict(optimizer="grid", n_points=n_points,
                                      diagonal_only=bool(K > 1)))


def optimize_fixed(obj, lo, hi, budget, rng, x0=None, value=2.0, **_):
    """No search at all: a constant radius. The tuned-once-and-frozen control."""
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    theta = clamp(np.full(lo.size, float(value)), lo, hi)
    J = safe_eval(obj, theta)
    return theta, make_info(obj, theta, J,
                            dict(optimizer="fixed", value=float(value)))


def optimize_nelder_mead(obj, lo, hi, budget, rng, x0=None, **_):
    """Derivative-free local search, as a single-trajectory reference.

    Included because the objective is smooth in the tapered parameterization,
    and a local method that starts in the right basin is hard to beat there.
    If it wins, that is a fact about the problem and the paper should report
    it rather than hide it behind a population.
    """
    from scipy.optimize import minimize

    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    start = (clamp(np.atleast_1d(np.asarray(x0, dtype=float)), lo, hi)
             if x0 is not None else rng.uniform(lo, hi))

    best = {"theta": start.copy(), "J": INF}

    def wrapped(theta):
        J = safe_eval(obj, clamp(theta, lo, hi))
        if J < best["J"]:
            best["theta"], best["J"] = clamp(theta, lo, hi).copy(), J
        return J if np.isfinite(J) else 1e12

    try:
        minimize(wrapped, start, method="Nelder-Mead",
                 options=dict(maxfev=int(budget) if budget else 200,
                              xatol=1e-3, fatol=1e-8))
    except Exception:
        pass
    return best["theta"], make_info(obj, best["theta"], best["J"],
                                    dict(optimizer="nelder-mead"))
