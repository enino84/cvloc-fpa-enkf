# -*- coding: utf-8 -*-
"""Common plumbing for the radius-selection optimizers.

Every optimizer exposes the same signature

    optimize(obj, lo, hi, budget, rng, x0=None, **params) -> (theta, info)

where ``obj`` is a :class:`cvloc.objective.CVObjective`, and ``lo`` and ``hi``
are the bounds of the admissible box, one entry per free radius. Optimizers
must stop when the budget runs out; the safe way is to wrap every call in
``safe_eval``, which turns a ``BudgetExhausted`` into a sentinel instead of an
exception in the middle of a population loop.
"""
from __future__ import annotations

import numpy as np

from ..objective import BudgetExhausted

INF = float("inf")


def safe_eval(obj, theta):
    """Evaluate J, returning +inf once the budget is gone."""
    try:
        return obj(theta)
    except BudgetExhausted:
        return INF


def out_of_budget(obj) -> bool:
    return obj.remaining() <= 0


def clamp(theta, lo, hi):
    return np.minimum(np.maximum(np.asarray(theta, dtype=float), lo), hi)


def init_population(size, lo, hi, rng, x0=None):
    """Uniform population in the box, optionally seeded with a warm start."""
    K = np.size(lo)
    pop = rng.uniform(lo, hi, size=(size, K))
    if x0 is not None:
        pop[0] = clamp(np.atleast_1d(np.asarray(x0, dtype=float)), lo, hi)
    return pop


def make_info(obj, theta_star, value_star, extra=None):
    info = dict(
        J=float(value_star),
        n_evals=int(obj.n_evals),
        n_calls=int(getattr(obj, "n_calls", obj.n_evals)),
        n_cache_hits=int(obj.n_cache_hits),
        stalled=bool(getattr(obj, "stalled", False)),
        theta=np.atleast_1d(np.asarray(theta_star, dtype=float)).tolist(),
    )
    if extra:
        info.update(extra)
    return info


def levy(K, lam, rng, scale=0.01):
    """Mantegna's algorithm for a symmetric Levy step with exponent ``lam``.

    ``scale`` is the 0.01 factor that is built into the reference Flower
    Pollination implementation. It is exposed separately from the step size
    ``gamma`` so that the two can be told apart: an implementation that omits
    it and then uses gamma = 1 is taking steps a hundred times larger than the
    reference one, which is the first thing to check when a run fails to
    converge.
    """
    from math import gamma as gamma_fn
    from math import pi, sin

    sigma = (gamma_fn(1.0 + lam) * sin(pi * lam / 2.0)
             / (gamma_fn((1.0 + lam) / 2.0) * lam * 2.0 ** ((lam - 1.0) / 2.0))
             ) ** (1.0 / lam)
    u = rng.standard_normal(K) * sigma
    v = rng.standard_normal(K)
    return scale * u / np.abs(v) ** (1.0 / lam)
