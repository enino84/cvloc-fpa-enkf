# -*- coding: utf-8 -*-
"""
Generation of assimilation cycles and the select-then-analyze driver.

The configuration the draft reports is a single cycle from a **climatological**
ensemble: members are widely separated snapshots of a long free run, so the
background error is large (RMSE near 3.7) and the sample covariance carries the
full sampling noise of a raw ensemble. That is the regime in which localization
is decisive and therefore the regime in which the criterion has something to
select. Building the ensemble any other way, for instance by perturbing the
truth, quietly removes most of what the radius is for.

The truth is an independent state on the same attractor, not a member and not
the ensemble mean displaced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ensemble_space import Cycle, EnsembleSpace
from .metaheuristics import get_optimizer
from .model import Lorenz96CV, forcing_profile, observation_network
from .objective import CVObjective
from .parameterization import box, make


@dataclass
class CycleConfig:
    """Everything that defines how a cycle is drawn."""
    n: int = 40
    N: int = 20
    F: float = 8.0
    dt: float = 0.01
    forcing: str = "uniform"
    F_lo: float = 5.0
    F_hi: float = 12.0
    n_regions: int = 4
    network: str = "full"
    dense_frac: float = 0.9
    sparse_frac: float = 0.3
    obs_std: float = 1.0
    obs_std_hi: float = None      # heterogeneous observation error, if set
    spinup: float = 20.0
    spacing: float = 1.0          # separation between climatological snapshots


def make_cycle(cfg: CycleConfig, seed: int) -> Cycle:
    """Draw one independent cycle."""
    rng = np.random.default_rng(seed)

    F = (cfg.F if cfg.forcing == "uniform" and np.isscalar(cfg.F)
         else forcing_profile(cfg.n, cfg.forcing, cfg.F_lo, cfg.F_hi,
                              cfg.n_regions))
    if cfg.forcing == "uniform":
        F = float(cfg.F)
    model = Lorenz96CV(n=cfg.n, F=F, dt=cfg.dt)

    # One long free run per cycle; the ensemble and the truth are taken from
    # well-separated points of it, so they are independent draws from the same
    # climatology rather than perturbations of one another.
    x = np.asarray(model.F_field, dtype=float).mean() * np.ones(cfg.n)
    x = x + 0.01 * rng.standard_normal(cfg.n)
    x = model.propagate(x, np.array([0.0, cfg.spinup]))

    snapshots = model.free_run(x, cfg.spacing, cfg.N + 1)
    Xb = snapshots[:, :cfg.N]
    x_true = snapshots[:, cfg.N]
    # Move the truth further along the attractor so it shares no snapshot with
    # the ensemble.
    x_true = model.propagate(x_true, np.array([0.0, 3.0 * cfg.spacing]))

    obs_idx = observation_network(cfg.n, cfg.network, rng,
                                  cfg.dense_frac, cfg.sparse_frac,
                                  cfg.n_regions)
    p = obs_idx.size

    if cfg.obs_std_hi is None:
        std = np.full(p, float(cfg.obs_std))
    else:
        block = (obs_idx * cfg.n_regions) // cfg.n
        std = np.where(block % 2 == 0, cfg.obs_std, cfg.obs_std_hi).astype(float)

    y = x_true[obs_idx] + std * rng.standard_normal(p)

    return Cycle(Xb=Xb, x_true=x_true, obs_idx=obs_idx, y=y,
                 r_inv=1.0 / std ** 2, model=model,
                 meta=dict(seed=int(seed), forcing=cfg.forcing,
                           network=cfg.network, N=int(cfg.N), n=int(cfg.n),
                           p=int(p)))


def make_cycles(cfg: CycleConfig, n_cycles: int, seed_base: int = 42) -> list:
    return [make_cycle(cfg, seed_base + i) for i in range(n_cycles)]


# ----------------------------------------------------------------------
def select_radius(space, optimizer="fpa", param_kind="uniform", K=1,
                  labels=None, box_name="informed", folds=10, budget=80,
                  seed=0, fold_seed=20260828, x0=None, opt_params=None,
                  quantize=6):
    """Select a radius on one cycle and report everything about the search.

    The fold seed is deliberately independent of the optimizer: two searches
    compared on the same cycle must minimize the *same* function, otherwise the
    comparison measures which noise realization each was handed.
    """
    param = make(param_kind, space.n, K=K, labels=labels)
    lo, hi = box(box_name, param.K)

    obj = CVObjective(space, param=param, folds=folds,
                      rng=np.random.default_rng(fold_seed),
                      budget=budget, quantize=quantize)

    optimize = get_optimizer(optimizer)
    params = dict(opt_params or {})
    theta, info = optimize(obj, lo, hi, budget, np.random.default_rng(seed),
                           x0=x0, **params)

    r = param.expand(theta)
    info.update(dict(param=param.label(), K=int(param.K), box=box_name,
                     radius_mean=float(np.mean(r)), radius_std=float(np.std(r))))
    return theta, r, obj, info


def select_radius_bagged(space, optimizer="fpa", param_kind="uniform", K=1,
                         labels=None, box_name="wide", folds=10, budget=80,
                         n_resamples=15, warm=True, seed=0,
                         fold_seed=20260828, opt_params=None, n_bins=24):
    """Select the radius by the *mode* of many resampled arg-minima.

    The estimator of :func:`select_radius` averages J over the folds and
    minimizes the average, which is the criterion as written in the draft. This
    one instead redraws the train/validation split ``B`` times, minimizes each
    realization separately, and reports the most frequent arg-min.

    The reason to prefer it is the flatness the draft itself reports: when the
    curve is nearly horizontal over a range of radii, the arg-min of one noisy
    realization moves a great deal even though the expected minimizer does not.
    A mode over B realizations only needs each vote to land in the right bin,
    not to hit the exact minimum, so it degrades far more gracefully. It also
    returns a distribution of radii rather than a point, which is an
    uncertainty interval at no extra cost.

    The observation network does **not** change between resamples; only the
    partition of it does. Nothing is recomputed and no model integration is
    performed, so B resamples cost B searches and not one cycle more.

    ``warm`` carries the population from one resample to the next. That is the
    one place where a population search has an advantage that does not depend
    on the shape of the landscape: a single-trajectory method has one incumbent
    to carry, a population carries a distribution over plausible radii and
    re-converges from it in a fraction of the budget.
    """
    param = make(param_kind, space.n, K=K, labels=labels)
    lo, hi = box(box_name, param.K)
    optimize = get_optimizer(optimizer)
    params = dict(opt_params or {})

    votes, total_evals, x0 = [], 0, None
    for b in range(int(n_resamples)):
        obj = CVObjective(space, param=param, folds=folds,
                          rng=np.random.default_rng(fold_seed + 104729 * b),
                          budget=budget)
        theta, info = optimize(obj, lo, hi, budget,
                               np.random.default_rng(seed + 7919 * b),
                               x0=(x0 if warm else None), **params)
        votes.append(np.atleast_1d(np.asarray(theta, dtype=float)).copy())
        total_evals += info["n_evals"]
        if warm:
            x0 = theta

    votes = np.array(votes)                       # (B, K)
    # Binning is logarithmic because the radius is a scale: the step from 1 to
    # 2 is the same kind of change as from 8 to 16.
    edges = np.geomspace(lo[0], hi[0], int(n_bins) + 1)
    theta_star = np.empty(param.K)
    for k in range(param.K):
        counts, _ = np.histogram(votes[:, k], bins=edges)
        i = int(counts.argmax())
        theta_star[k] = float(np.sqrt(edges[i] * edges[i + 1]))

    r = param.expand(theta_star)
    info = dict(optimizer=f"{optimizer}-bagged", J=float("nan"),
                n_evals=total_evals, n_calls=total_evals, n_cache_hits=0,
                stalled=False, param=param.label(), K=int(param.K),
                box=box_name, n_resamples=int(n_resamples), warm=bool(warm),
                radius_mean=float(np.mean(r)), radius_std=float(np.std(r)),
                vote_iqr=float(np.subtract(*np.percentile(votes[:, 0], [75, 25]))),
                theta=theta_star.tolist())
    return theta_star, r, votes, info


def truth_optimum(space, lo=0.2, hi=20.0, n_points=200, log_spaced=True):
    """Dense sweep of the uniform radius against the truth.

    This is the oracle an operational system could not compute. It is the
    reference for the penalty reported in EXP-03, and the sweep it produces is
    the error curve of EXP-01.
    """
    grid = (np.geomspace(lo, hi, n_points) if log_spaced
            else np.linspace(lo, hi, n_points))
    err = np.array([space.truth_error(np.full(space.n, r)) for r in grid])
    i = int(np.argmin(err))
    return dict(grid=grid, error=err, r_opt=float(grid[i]),
                error_opt=float(err[i]), error_worst=float(err.max()))


def cv_sweep(space, folds=10, fold_seed=20260828, lo=0.2, hi=20.0,
             n_points=200, log_spaced=True):
    """Dense sweep of the uniform radius against the cross-validated cost."""
    grid = (np.geomspace(lo, hi, n_points) if log_spaced
            else np.linspace(lo, hi, n_points))
    obj = CVObjective(space, param=make("uniform", space.n), folds=folds,
                      rng=np.random.default_rng(fold_seed), budget=None)
    cost = np.array([obj.raw(np.array([r])) for r in grid])
    i = int(np.argmin(cost))
    return dict(grid=grid, cost=cost, r_opt=float(grid[i]),
                cost_opt=float(cost[i]))
