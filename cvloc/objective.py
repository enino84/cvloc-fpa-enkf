# -*- coding: utf-8 -*-
"""
The cross-validated criterion.

Let O be the observations available at a cycle and O = T_i u V_i, i = 1..m, a
k-fold partition. With ``x_a(r; T_i)`` the analysis computed from the training
subset alone,

    J(r) = (1/m) sum_i  || y_Vi - H_Vi x_a(r; T_i) ||^2_{R_Vi^{-1}} / |V_i|

No reference trajectory enters J, so it is computable in an operational
setting. Two implementation points are not optional:

* The m partitions are drawn **once per cycle and frozen** across every
  candidate evaluated within that cycle. Resampling per candidate would rank
  different radii on different noise realizations, and the ranking, not the
  level, is what the search consumes.
* J contains the observation error variance of the validation set, which does
  not depend on r. It does not move the minimizer but it raises a floor that
  flattens the landscape, so averaging over folds is part of the estimator and
  not a refinement.

Budget accounting mirrors the objective of the companion suite: every optimizer
receives the same object, unique evaluations are counted, repeats are cached
and counted separately. Two metaheuristics compared at "equal budget" are then
comparable in the only sense that matters, namely how many times they were
allowed to look at the function.
"""
from __future__ import annotations

import numpy as np

from .parameterization import Uniform


class BudgetExhausted(Exception):
    """Raised when an optimizer asks for one evaluation too many."""


def make_folds(p: int, m: int, rng) -> list:
    """Partition the observation indices into m folds."""
    m = int(min(m, p))
    perm = rng.permutation(p)
    return [np.sort(part) for part in np.array_split(perm, m)]


class CVObjective:
    """J(r) evaluated on frozen folds of the observation network.

    Parameters
    ----------
    space : EnsembleSpace
        The cycle, with everything that does not depend on r precomputed.
    param : Parameterization
        Map from theta in R^K to one radius per analysis point.
    folds : int
        Number of folds m.
    rng : Generator
        Used once, to draw the folds. Not consulted afterwards.
    budget : int or None
        Maximum number of unique evaluations. ``None`` means unlimited, which
        is what the exhaustive sweeps use.
    quantize : int or None
        Decimals used to build the cache key. Distinct candidates that agree to
        this precision share an evaluation. Set to ``None`` for a discrete
        parameterization whose values are already integers.
    """

    def __init__(self, space, param=None, folds=10, rng=None, budget=None,
                 quantize=6, call_multiplier=25):
        self.space = space
        self.param = param if param is not None else Uniform(space.n)
        self.K = self.param.K
        self.budget = None if budget is None else int(budget)
        self.quantize = quantize
        self.call_limit = (None if self.budget is None
                           else int(self.budget * call_multiplier))

        rng = np.random.default_rng(0) if rng is None else rng
        p = space.cycle.p
        self._folds = make_folds(p, folds, rng)
        self.n_folds = len(self._folds)

        # Per fold: the training mask (1 on training rows, 0 on validation) and
        # the validation rows, with their observation precisions.
        self._blocks = []
        for V in self._folds:
            mask = np.ones(p)
            mask[V] = 0.0
            self._blocks.append(dict(
                mask=mask,
                V=V,
                y_V=np.asarray(space.cycle.y)[V],
                ri_V=space.r_inv[V],
                idx_V=space.obs_idx[V],
            ))

        self.n_evals = 0
        self.n_calls = 0
        self.n_cache_hits = 0
        self.stalled = False
        self._cache: dict = {}
        self.trace: list = []
        self.arg_trace: list = []

    # ------------------------------------------------------------------
    def _key(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if self.quantize is None:
            return tuple(int(round(v)) for v in theta)
        return tuple(np.round(theta, self.quantize))

    def raw(self, theta) -> float:
        """Evaluate J without touching the budget or the cache."""
        r = self.param.expand(theta)
        total = 0.0
        for blk in self._blocks:
            xa = self.space.analysis_mean(r, mask=blk["mask"])
            resid = blk["y_V"] - xa[blk["idx_V"]]
            total += float(resid @ (blk["ri_V"] * resid)) / resid.size
        return total / self.n_folds

    def __call__(self, theta) -> float:
        self.n_calls += 1
        if self.call_limit is not None and self.n_calls > self.call_limit:
            self.stalled = True
            raise BudgetExhausted(
                f"call limit of {self.call_limit} reached "
                f"({self.n_evals} unique evaluations)")

        key = self._key(theta)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        if self.budget is not None and self.n_evals >= self.budget:
            raise BudgetExhausted(f"budget of {self.budget} evaluations used up")

        value = self.raw(theta)
        self._cache[key] = value
        self.n_evals += 1
        self.trace.append(value)
        self.arg_trace.append(np.atleast_1d(np.asarray(theta, dtype=float)).copy())
        return value

    # ------------------------------------------------------------------
    def remaining(self) -> float:
        if self.budget is None:
            return float("inf")
        if self.call_limit is not None and self.n_calls >= self.call_limit:
            return 0
        return max(0, self.budget - self.n_evals)

    def best_so_far(self) -> float:
        return min(self._cache.values()) if self._cache else float("nan")

    def best_trace(self) -> np.ndarray:
        """Running minimum of J over unique evaluations."""
        if not self.trace:
            return np.empty(0)
        return np.minimum.accumulate(np.asarray(self.trace, dtype=float))

    def fork(self, budget=None, param=None):
        """A copy sharing the frozen folds, with an empty cache.

        This is what makes a fair head-to-head possible: every optimizer
        minimizes the *same* function, over the *same* folds, from the *same*
        ensemble, with its own budget and its own cache.
        """
        other = object.__new__(CVObjective)
        other.__dict__.update(self.__dict__)
        if param is not None:
            other.param = param
            other.K = param.K
        other.budget = self.budget if budget is None else int(budget)
        other.call_limit = (None if other.budget is None
                            else int(other.budget * 25))
        other.n_evals = 0
        other.n_calls = 0
        other.n_cache_hits = 0
        other.stalled = False
        other._cache = {}
        other.trace = []
        other.arg_trace = []
        return other


class TruthObjective:
    """The oracle: RMSE of the analysis against the true state.

    Never used to select anything a method reports. It bounds what selection
    could achieve and gives the reference against which the penalty of the
    cross-validated choice is measured.
    """

    def __init__(self, space, param=None):
        self.space = space
        self.param = param if param is not None else Uniform(space.n)
        self.K = self.param.K

    def __call__(self, theta) -> float:
        return self.space.truth_error(self.param.expand(theta))
