# -*- coding: utf-8 -*-
"""Simulated annealing over the admissible box.

The single-trajectory scheme the draft replaced. It is kept as a competitor
rather than deleted, because the claim that a population is preferable under
sampling noise is a claim about a comparison, and the comparison needs the
other side to be present and fairly tuned.

The initial temperature is calibrated from a short random walk so that a
typical worsening move is accepted with probability about 0.8, which puts SA
and FPA on the same footing: neither gets a hand-tuned constant the other
does not.
"""
from __future__ import annotations

import numpy as np

from .base import INF, clamp, make_info, out_of_budget, safe_eval


def optimize(obj, lo, hi, budget, rng, x0=None,
             T0=None, cooling=0.92, level_iters=None, T_min=1e-10,
             step_frac=0.15, step_decay=0.98, calib_moves=8, **_):
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    K = lo.size
    width = hi - lo

    theta = (clamp(np.atleast_1d(np.asarray(x0, dtype=float)), lo, hi)
             if x0 is not None else rng.uniform(lo, hi))
    J = safe_eval(obj, theta)
    theta_star, J_star = theta.copy(), J

    if level_iters is None:
        level_iters = max(2 * K, 10)

    sigma = step_frac * width

    if T0 is None:
        deltas = []
        probe, probe_J = theta.copy(), J
        for _ in range(calib_moves):
            if out_of_budget(obj):
                break
            cand = clamp(probe + sigma * rng.standard_normal(K), lo, hi)
            cand_J = safe_eval(obj, cand)
            if np.isfinite(cand_J) and cand_J > probe_J:
                deltas.append(cand_J - probe_J)
            if np.isfinite(cand_J):
                probe, probe_J = cand, cand_J
                if cand_J < J_star:
                    theta_star, J_star = cand.copy(), cand_J
        mean_delta = float(np.mean(deltas)) if deltas else max(abs(J) * 1e-2, 1e-9)
        T0 = -mean_delta / np.log(0.8)

    T = float(T0)
    n_accept = n_uphill = 0
    while not out_of_budget(obj) and T > T_min:
        for _ in range(level_iters):
            if out_of_budget(obj):
                break
            cand = clamp(theta + sigma * rng.standard_normal(K), lo, hi)
            cand_J = safe_eval(obj, cand)
            if cand_J == INF:
                break
            delta = cand_J - J
            if delta <= 0.0:
                theta, J = cand, cand_J
                n_accept += 1
            elif rng.random() < np.exp(-delta / T):
                theta, J = cand, cand_J
                n_accept += 1
                n_uphill += 1
            if J < J_star:
                theta_star, J_star = theta.copy(), J
        T *= cooling
        sigma = sigma * step_decay

    return theta_star, make_info(obj, theta_star, J_star,
                                 dict(optimizer="sa", T0=float(T0),
                                      n_accept=n_accept, n_uphill=n_uphill))
