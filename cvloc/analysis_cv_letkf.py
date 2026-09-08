# -*- coding: utf-8 -*-
"""
CV-LETKF: an R-localized LETKF whose radius is re-estimated at every cycle by
minimizing the cross-validated criterion.

Registers itself into pyteda's global analysis registry as ``letkf-cv``, so
importing :mod:`cvloc` is enough to use it from ``AnalysisFactory``,
``Simulation.from_scenario`` and ``Benchmark``:

    import cvloc
    cfg = dict(method="letkf-cv", optimizer="fpa", budget=80, warm_start=True)

This is the class EXP-09 drives. In the single-cycle experiments the driver in
:mod:`cvloc.cycles` is used instead, because there the filter loop would only
get in the way.

Warm start
----------
Across consecutive cycles the radius is a slowly varying quantity: the ensemble
changes, but not so much that yesterday's radius is uninformative today.
Carrying the population forward is therefore not a convenience, it is the one
place where a population search has a structural advantage over a single
trajectory: the population encodes a distribution over plausible radii, and a
single incumbent does not. ``warm_start`` seeds the next cycle's search at the
previous solution, and ``carry_population`` additionally reuses the final
population as the initial one.

Every cycle appends a record to ``self.radius_history`` with the selected
radius field, the value of J attained, the budget actually spent and the wall
time of the search, so a run can be audited without rerunning it.
"""
from __future__ import annotations

import time

import numpy as np
from pyteda.analysis.analysis_core import Analysis
from pyteda.analysis.registry import register_analysis

from .ensemble_space import Cycle, EnsembleSpace
from .metaheuristics import defaults_for, get_optimizer
from .objective import CVObjective
from .parameterization import box, make


@register_analysis("letkf-cv")
class AnalysisLETKFCrossValidated(Analysis):
    """R-localized LETKF with a cross-validated localization radius.

    Parameters
    ----------
    model : pyteda model on a ring, with ``get_number_of_variables``.
    optimizer : str
        Any key of :data:`cvloc.metaheuristics.OPTIMIZERS`.
    param_kind, K : str, int
        Radius parameterization and its number of free parameters.
    box_name : str
        Admissible box, see :data:`cvloc.parameterization.BOXES`.
    folds : int
        Number of observation folds.
    budget : int
        Unique objective evaluations allowed per cycle.
    warm_start : bool
        Start the search from the previous cycle's radius.
    carry_population : bool
        Reuse the previous population as the initial one. Requires an optimizer
        that accepts ``init_pop``; ignored otherwise.
    fixed_radius : float or None
        If set, no search is done and this radius is used at every cycle. The
        tuned-once-and-frozen control.
    smoothing : str
        How the per-cycle estimate is turned into the radius actually used.
        ``'none'`` uses the raw estimate, which is what the draft describes.
        ``'ewma'`` applies exponential smoothing with factor ``alpha``.
        ``'window'`` averages the last ``window`` estimates in log space.
        ``'freeze'`` estimates on the first ``freeze_after`` cycles and then
        holds the result fixed.

        This matters more than it looks. The criterion identifies the right
        radius on average, but the landscape is flat near its minimum, so the
        arg-min of one noisy realization moves a great deal from cycle to
        cycle even when the quantity being estimated is nearly constant. The
        analysis pays for every one of those jumps. Treating the per-cycle
        estimate as the parameter is therefore a mistake, and the radius has to
        be handled as what it is: a noisy estimate of a slowly varying
        quantity.
    alpha : float
        Smoothing factor for ``'ewma'``. Small means slow and stable.
    window : int
        Number of cycles averaged by ``'window'``.
    freeze_after : int
        Number of cycles estimated before freezing, for ``'freeze'``.
    """

    def __init__(self, model, optimizer="fpa", param_kind="uniform", K=1,
                 box_name="informed", folds=10, budget=80, seed=0,
                 fold_seed=20260828, warm_start=True, carry_population=False,
                 fixed_radius=None, smoothing="none", alpha=0.2, window=5,
                 freeze_after=5, store_states=True, opt_params=None,
                 tag=None, **kwargs):
        self.model = model
        self.n = int(model.get_number_of_variables())
        self.optimizer_name = str(optimizer)
        self.param_kind = str(param_kind)
        self.K = int(K)
        self.box_name = str(box_name)
        self.folds = int(folds)
        self.budget = int(budget)
        self.seed = int(seed)
        self.fold_seed = int(fold_seed)
        self.warm_start = bool(warm_start)
        self.carry_population = bool(carry_population)
        self.fixed_radius = fixed_radius
        self.smoothing = str(smoothing)
        self.alpha = float(alpha)
        self.window = int(window)
        self.freeze_after = int(freeze_after)
        self._raw_history: list = []
        self._frozen_theta = None
        # pyteda keeps per-cycle states only in its legacy mode, so the means
        # are recorded here instead. Two vectors of n floats per cycle is a few
        # hundred kilobytes over a long run, which is cheap enough to keep
        # always and is what every time series in the paper is drawn from. The
        # full ensembles are a separate matter and come from the simulation's
        # own snapshot machinery.
        self.store_states = bool(store_states)
        self.xb_history: list = []
        self.xa_history: list = []
        self.opt_params = {**defaults_for(optimizer), **dict(opt_params or {})}
        self.tag = tag if tag is not None else f"cv-{optimizer}"

        self._optimize = get_optimizer(self.optimizer_name)
        self._param = make(self.param_kind, self.n, K=self.K)
        self._lo, self._hi = box(self.box_name, self._param.K)
        self._rng = np.random.default_rng(self.seed)
        self._prev_theta = None

        self.radius_history: list = []
        self.cycle = 0

    # ------------------------------------------------------------------
    @property
    def state_means(self):
        """(background means, analysis means), each (n_cycles, n_state)."""
        if not self.xa_history:
            return np.empty((0, self.n)), np.empty((0, self.n))
        return np.stack(self.xb_history), np.stack(self.xa_history)

    @property
    def radius_profile(self) -> np.ndarray:
        """(n_cycles, n) array of the radius field used at every cycle."""
        if not self.radius_history:
            return np.empty((0, self.n))
        return np.stack([rec["r"] for rec in self.radius_history])

    # ------------------------------------------------------------------
    def perform_assimilation(self, background, observation):
        Xb = background.get_ensemble()
        y = np.asarray(observation.get_observation(), dtype=float)
        obs_idx = np.asarray(observation.H_index, dtype=int)

        noise = getattr(observation, "noise", None)
        if noise is not None and hasattr(noise, "R_inv_diag"):
            r_inv = np.asarray(noise.R_inv_diag, dtype=float)
        else:
            R = np.asarray(observation.get_data_error_covariance())
            r_inv = 1.0 / np.diag(R)

        cycle = Cycle(Xb=Xb, x_true=np.zeros(Xb.shape[0]), obs_idx=obs_idx,
                      y=y, r_inv=r_inv, model=self.model)
        space = EnsembleSpace(cycle)

        t0 = time.perf_counter()
        if self.fixed_radius is not None:
            theta = np.full(self._param.K, float(self.fixed_radius))
            obj = CVObjective(space, param=self._param, folds=self.folds,
                              rng=np.random.default_rng(
                                  self.fold_seed + 7919 * self.cycle),
                              budget=None)
            info = dict(J=float(obj.raw(theta)), n_evals=0, n_calls=0,
                        n_cache_hits=0, stalled=False, optimizer="fixed")
        else:
            obj = CVObjective(space, param=self._param, folds=self.folds,
                              rng=np.random.default_rng(
                                  self.fold_seed + 7919 * self.cycle),
                              budget=self.budget)
            x0 = self._prev_theta if (self.warm_start and
                                      self._prev_theta is not None) else None
            theta, info = self._optimize(obj, self._lo, self._hi, self.budget,
                                         self._rng, x0=x0, **self.opt_params)
        t_search = time.perf_counter() - t0

        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        self._prev_theta = theta.copy()
        theta_raw = theta.copy()
        theta = self._smooth(theta_raw)
        r = self._param.expand(theta)

        self.radius_history.append(dict(
            cycle=self.cycle, tag=self.tag, optimizer=self.optimizer_name,
            theta=theta.copy(), theta_raw=theta_raw.copy(), r=r.copy(),
            radius_raw=float(np.mean(self._param.expand(theta_raw))),
            radius_mean=float(np.mean(r)), radius_std=float(np.std(r)),
            J=float(info.get("J", np.nan)),
            n_evals=int(info.get("n_evals", 0)),
            n_calls=int(info.get("n_calls", 0)),
            n_cache_hits=int(info.get("n_cache_hits", 0)),
            stalled=bool(info.get("stalled", False)),
            time_search=float(t_search),
        ))
        self.cycle += 1

        _, Xa = space.analysis_ensemble(r)
        self.Xa = Xa
        if self.store_states:
            self.xb_history.append(space.xb.astype(np.float32))
            self.xa_history.append(Xa.mean(axis=1).astype(np.float32))
        return self.Xa

    # ------------------------------------------------------------------
    def _smooth(self, theta_raw):
        """Turn the raw per-cycle estimate into the radius actually used.

        Smoothing is done in log space, because the radius is a scale: the
        distance from 1 to 2 is the same kind of change as from 4 to 8, and an
        arithmetic mean of estimates that straddle an order of magnitude is
        dominated by the largest of them.
        """
        self._raw_history.append(np.asarray(theta_raw, dtype=float).copy())
        mode = self.smoothing

        if mode == "none":
            return theta_raw
        if mode == "freeze":
            if len(self._raw_history) <= self.freeze_after:
                self._frozen_theta = np.exp(
                    np.mean(np.log(np.maximum(self._raw_history, 1e-12)), axis=0))
            return self._frozen_theta.copy()
        if mode == "window":
            recent = self._raw_history[-self.window:]
            return np.exp(np.mean(np.log(np.maximum(recent, 1e-12)), axis=0))
        if mode == "ewma":
            if len(self._raw_history) == 1:
                self._ewma = np.log(np.maximum(theta_raw, 1e-12))
            else:
                self._ewma = ((1.0 - self.alpha) * self._ewma
                              + self.alpha * np.log(np.maximum(theta_raw, 1e-12)))
            return np.exp(self._ewma)
        raise ValueError(f"unknown smoothing '{mode}'")

    def get_analysis_state(self):
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        return self.Xa

    def get_error_covariance(self):
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        _, N = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(N))
        self.Xa = np.outer(xa, np.ones(N)) + inflation_factor * DXa
