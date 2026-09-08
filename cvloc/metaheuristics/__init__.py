# -*- coding: utf-8 -*-
"""Registry of radius-selection optimizers.

All share the signature

    optimize(obj, lo, hi, budget, rng, x0=None, **params) -> (theta, info)
"""
from __future__ import annotations

from . import baselines, fpa, sa, swarm

OPTIMIZERS = {
    "fpa": fpa.optimize,
    "sa": sa.optimize,
    "pso": swarm.optimize_pso,
    "de": swarm.optimize_de,
    "ga": swarm.optimize_ga,
    "firefly": swarm.optimize_firefly,
    "random": baselines.optimize_random,
    "grid": baselines.optimize_grid,
    "fixed": baselines.optimize_fixed,
    "nelder-mead": baselines.optimize_nelder_mead,
}

LABELS = {
    "fpa": "Flower pollination",
    "sa": "Simulated annealing",
    "pso": "Particle swarm",
    "de": "Differential evolution",
    "ga": "Genetic algorithm",
    "firefly": "Firefly",
    "random": "Random sampling",
    "grid": "Grid sweep",
    "fixed": "Fixed radius",
    "nelder-mead": "Nelder-Mead",
}

# The population searches that are compared head to head.
METAHEURISTICS = ["fpa", "sa", "pso", "de", "ga", "firefly"]

# The controls that calibrate that comparison.
CONTROLS = ["random", "grid", "nelder-mead"]

# Default hyperparameters. EXP-00 overwrites these with the values it
# calibrates against J on held-out cycles, and writes them to
# results/frozen_params.json; every later experiment reads that file if it
# exists, so no number in the paper depends on a hand-tuned constant.
DEFAULTS = {
    "fpa": dict(pop_size=10, switch_p=0.8, lam=1.5, gamma=0.1,
                levy_scale=0.01, broadcast=False),
    "sa": dict(cooling=0.92, step_frac=0.15, step_decay=0.98),
    "pso": dict(pop_size=10, w=0.72, c1=1.49, c2=1.49),
    "de": dict(pop_size=10, F=0.6, CR=0.9),
    "ga": dict(pop_size=10, blx_alpha=0.5, p_mut=0.2, mut_frac=0.1),
    "firefly": dict(pop_size=10, beta0=1.0, absorb=1.0, zeta=0.2),
    "random": dict(),
    "grid": dict(log_spaced=True),
    "fixed": dict(value=2.0),
    "nelder-mead": dict(),
}


def get_optimizer(name):
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer '{name}'. "
                         f"Available: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name]


def defaults_for(name) -> dict:
    return dict(DEFAULTS.get(name, {}))


__all__ = ["OPTIMIZERS", "LABELS", "METAHEURISTICS", "CONTROLS", "DEFAULTS",
           "get_optimizer", "defaults_for"]
