# -*- coding: utf-8 -*-
"""
The other population searches: PSO, differential evolution, a real-coded
genetic algorithm and the firefly algorithm.

They exist so that the comparison in EXP-04 and EXP-10 has more than one side.
All four use the same population size and the same budget accounting as FPA,
and none of them gets a hyperparameter that was tuned against the truth: like
FPA, their parameters are calibrated in EXP-00 against J on held-out cycles.
"""
from __future__ import annotations

import numpy as np

from .base import (INF, clamp, init_population, make_info, out_of_budget,
                   safe_eval)


# ----------------------------------------------------------------------
def optimize_pso(obj, lo, hi, budget, rng, x0=None,
                 pop_size=10, w=0.72, c1=1.49, c2=1.49, vmax_frac=0.3, **_):
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    K, width = lo.size, hi - lo

    pos = init_population(pop_size, lo, hi, rng, x0=x0)
    vmax = vmax_frac * width
    vel = rng.uniform(-vmax, vmax, size=(pop_size, K))

    fit = np.array([safe_eval(obj, pos[s]) for s in range(pop_size)])
    pbest, pbest_f = pos.copy(), fit.copy()
    g = int(np.argmin(fit))
    theta_star, J_star = pos[g].copy(), float(fit[g])

    n_iters = 0
    while not out_of_budget(obj):
        n_iters += 1
        for s in range(pop_size):
            if out_of_budget(obj):
                break
            r1, r2 = rng.random(K), rng.random(K)
            vel[s] = (w * vel[s] + c1 * r1 * (pbest[s] - pos[s])
                      + c2 * r2 * (theta_star - pos[s]))
            vel[s] = np.clip(vel[s], -vmax, vmax)
            pos[s] = clamp(pos[s] + vel[s], lo, hi)
            f = safe_eval(obj, pos[s])
            if f == INF:
                break
            if f < pbest_f[s]:
                pbest[s], pbest_f[s] = pos[s].copy(), f
            if f < J_star:
                theta_star, J_star = pos[s].copy(), f

    return theta_star, make_info(obj, theta_star, J_star,
                                 dict(optimizer="pso", pop_size=int(pop_size),
                                      n_iters=n_iters))


# ----------------------------------------------------------------------
def optimize_de(obj, lo, hi, budget, rng, x0=None,
                pop_size=10, F=0.6, CR=0.9, **_):
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    K = lo.size
    pop_size = max(4, int(pop_size))

    pop = init_population(pop_size, lo, hi, rng, x0=x0)
    fit = np.array([safe_eval(obj, pop[s]) for s in range(pop_size)])
    g = int(np.argmin(fit))
    theta_star, J_star = pop[g].copy(), float(fit[g])

    n_iters = 0
    while not out_of_budget(obj):
        n_iters += 1
        for s in range(pop_size):
            if out_of_budget(obj):
                break
            a, b, c = rng.choice([i for i in range(pop_size) if i != s],
                                 size=3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            cross = rng.random(K) < CR
            if not cross.any():
                cross[rng.integers(K)] = True
            trial = clamp(np.where(cross, mutant, pop[s]), lo, hi)
            f = safe_eval(obj, trial)
            if f == INF:
                break
            if f <= fit[s]:
                pop[s], fit[s] = trial, f
            if f < J_star:
                theta_star, J_star = trial.copy(), f

    return theta_star, make_info(obj, theta_star, J_star,
                                 dict(optimizer="de", pop_size=int(pop_size),
                                      n_iters=n_iters))


# ----------------------------------------------------------------------
def optimize_ga(obj, lo, hi, budget, rng, x0=None,
                pop_size=10, blx_alpha=0.5, p_mut=0.2, mut_frac=0.1,
                elite=1, tournament=2, **_):
    """Real-coded GA with BLX-alpha crossover and gaussian mutation."""
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    K, width = lo.size, hi - lo

    pop = init_population(pop_size, lo, hi, rng, x0=x0)
    fit = np.array([safe_eval(obj, pop[s]) for s in range(pop_size)])
    g = int(np.argmin(fit))
    theta_star, J_star = pop[g].copy(), float(fit[g])

    n_iters = 0
    while not out_of_budget(obj):
        n_iters += 1
        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:elite]]
        new_fit = [fit[i] for i in order[:elite]]
        while len(new_pop) < pop_size:
            if out_of_budget(obj):
                break
            parents = []
            for _ in range(2):
                cand = rng.choice(pop_size, size=tournament, replace=False)
                parents.append(pop[cand[int(np.argmin(fit[cand]))]])
            p1, p2 = parents
            span = np.abs(p1 - p2)
            child = rng.uniform(np.minimum(p1, p2) - blx_alpha * span,
                                np.maximum(p1, p2) + blx_alpha * span)
            if rng.random() < p_mut:
                child = child + mut_frac * width * rng.standard_normal(K)
            child = clamp(child, lo, hi)
            f = safe_eval(obj, child)
            if f == INF:
                break
            new_pop.append(child)
            new_fit.append(f)
            if f < J_star:
                theta_star, J_star = child.copy(), f
        if len(new_pop) < pop_size:
            break
        pop = np.array(new_pop)
        fit = np.array(new_fit)

    return theta_star, make_info(obj, theta_star, J_star,
                                 dict(optimizer="ga", pop_size=int(pop_size),
                                      n_iters=n_iters))


# ----------------------------------------------------------------------
def optimize_firefly(obj, lo, hi, budget, rng, x0=None,
                     pop_size=10, beta0=1.0, absorb=1.0, zeta=0.2,
                     zeta_decay=0.97, **_):
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    K, width = lo.size, hi - lo

    pos = init_population(pop_size, lo, hi, rng, x0=x0)
    fit = np.array([safe_eval(obj, pos[s]) for s in range(pop_size)])
    g = int(np.argmin(fit))
    theta_star, J_star = pos[g].copy(), float(fit[g])

    z = float(zeta)
    n_iters = 0
    while not out_of_budget(obj):
        n_iters += 1
        for u in range(pop_size):
            for v in range(pop_size):
                if fit[v] >= fit[u]:
                    continue
                scaled = (pos[u] - pos[v]) / np.maximum(width, 1e-12)
                beta = beta0 * np.exp(-absorb * float(np.sum(scaled ** 2)))
                jitter = z * width * (rng.random(K) - 0.5)
                pos[u] = clamp(pos[u] + beta * (pos[v] - pos[u]) + jitter, lo, hi)
            if out_of_budget(obj):
                break
            f = safe_eval(obj, pos[u])
            if f == INF:
                break
            fit[u] = f
            if f < J_star:
                theta_star, J_star = pos[u].copy(), f
        z *= zeta_decay

    return theta_star, make_info(obj, theta_star, J_star,
                                 dict(optimizer="firefly",
                                      pop_size=int(pop_size), n_iters=n_iters))
