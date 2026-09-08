# -*- coding: utf-8 -*-
"""
Flower Pollination Algorithm.

The implementation follows Yang's reference code rather than the equation as
written in the draft, and the difference is worth stating because it is the
first thing a reader familiar with FPA will check.

**Direction of the global step.** The reference implementation computes

    dS = gamma * L(lambda) * (x_i - x_best)

while the draft writes ``r + gamma L(lambda) (r_best - r)``. Mantegna's step is
symmetric, so the two agree in distribution and neither is wrong; the code
below uses the reference form and exposes ``toward_best`` for the other.

**Step size.** ``L(lambda)`` in the reference code already carries a factor of
0.01 (see :func:`cvloc.metaheuristics.base.levy`). The effective step is
therefore ``gamma * 0.01 * step``. An implementation that drops that factor and
uses ``gamma = 1`` takes steps a hundred times larger than intended, which
late in the run means the population never settles. ``levy_scale`` and
``gamma`` are kept separate here so the two can be varied independently and
the effect attributed to one or the other; EXP-00 sweeps ``gamma`` over
0.01 to 1.0 for exactly this reason.

**Switching probability.** The reference code performs the *global* Levy step
when ``rand > p`` and local pollination otherwise, so ``p = 0.8`` means global
pollination one time in five, not four. The draft's phrasing invites the
opposite reading. ``switch_is_global`` makes the convention explicit.

**Broadcast.** Yang's suggestion for the multi-radius case: the best value
found among all components should be shared with the others, so candidates are
perturbed around it in different regions instead of each component searching on
its own. ``broadcast`` implements this as an occasional move that resets every component
of a candidate to a single shared reference plus a perturbation. The objective
returns one number for the whole radius field, so there is no per-component
score to rank by and the reference used is the median of the incumbent best.
The move is off by default so that EXP-07 can switch it on and measure what it
is worth, rather than having it baked in.
"""
from __future__ import annotations

import numpy as np

from .base import (INF, clamp, init_population, levy, make_info, out_of_budget,
                   safe_eval)


def optimize(obj, lo, hi, budget, rng, x0=None,
             pop_size=10, switch_p=0.8, lam=1.5, gamma=0.1,
             levy_scale=0.01, switch_is_global=False, toward_best=False,
             broadcast=False, broadcast_p=0.15, broadcast_sigma=0.15, **_):
    """Minimize ``obj`` over the box ``[lo, hi]``.

    Parameters
    ----------
    pop_size : int
        Population size S.
    switch_p : float
        Switching probability p. With ``switch_is_global=False`` (the
        reference convention) a candidate takes the global Levy step when
        ``rand > p``.
    lam : float
        Levy exponent.
    gamma : float
        Scaling of the global step, on top of ``levy_scale``.
    broadcast : bool
        Share the best component value across components. Only meaningful when
        there is more than one free radius.
    """
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    K = lo.size

    pop = init_population(pop_size, lo, hi, rng, x0=x0)
    fit = np.empty(pop_size)
    for s in range(pop_size):
        fit[s] = safe_eval(obj, pop[s])

    best = int(np.argmin(fit))
    theta_star, J_star = pop[best].copy(), float(fit[best])

    n_global = n_local = n_broadcast = n_iters = 0
    while not out_of_budget(obj):
        n_iters += 1
        for s in range(pop_size):
            if out_of_budget(obj):
                break

            u = rng.random()
            global_step = (u < switch_p) if switch_is_global else (u > switch_p)

            if broadcast and K > 1 and rng.random() < broadcast_p:
                # Yang's suggestion: take the best radius found so far and
                # perturb every component around it, so the information gained
                # about one region is not thrown away in the others.
                r_ref = float(np.median(theta_star))
                spread = broadcast_sigma * (hi - lo)
                cand = r_ref + spread * rng.standard_normal(K)
                n_broadcast += 1
            elif global_step:
                L = levy(K, lam, rng, scale=levy_scale)
                step = (theta_star - pop[s]) if toward_best else (pop[s] - theta_star)
                cand = pop[s] + gamma * L * step
                n_global += 1
            else:
                u_idx, v_idx = rng.choice(pop_size, size=2, replace=False)
                eps = rng.random()
                cand = pop[s] + eps * (pop[u_idx] - pop[v_idx])
                n_local += 1

            cand = clamp(cand, lo, hi)
            J = safe_eval(obj, cand)
            if J == INF:
                break
            if J < fit[s]:
                pop[s], fit[s] = cand, J
            if J < J_star:
                theta_star, J_star = cand.copy(), J

    return theta_star, make_info(
        obj, theta_star, J_star,
        dict(optimizer="fpa", pop_size=int(pop_size), switch_p=float(switch_p),
             lam=float(lam), gamma=float(gamma), levy_scale=float(levy_scale),
             broadcast=bool(broadcast), n_iters=n_iters,
             n_global=n_global, n_local=n_local, n_broadcast=n_broadcast))
