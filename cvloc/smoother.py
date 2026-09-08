# -*- coding: utf-8 -*-
"""
EnKF smoother on the augmented state (k-1, k), with a modified Cholesky
precision, and a localization radius selected by a criterion that uses
observations only.

Why this formulation
--------------------
Everything that came before it failed for a reason that is worth stating,
because it is the argument for this design.

Scoring the analysis against the observations it assimilated is circular: a
short radius makes each grid point fit its own observation, the residual goes
to zero, and the criterion is minimized at the left edge of the box. Measured
directly, J falls monotonically to r = 0.2.

Holding observations out fixes the circularity but creates a worse problem. The
analysis at a held-out location depends only on the radius *at that location*,
so with p < n the radii of unobserved points do not enter the objective at all.
Measured directly: moving the twenty radii of unobserved points from 3 to 15
left J unchanged to eight decimal places while the true error nearly doubled.
Half the parameters were invisible to the criterion.

Scoring a forecast against future observations would fix that, but at cycle k
the observations of k+1 do not exist yet.

This formulation resolves all three. The state is the pair (k-1, k). The
modified Cholesky factorization is estimated on the joint vector, so its
cross-block couples every component of k-1 to the components of k. Only the
observations of k-1 are assimilated. The radius is then scored on the k block
against the observations of k, which that analysis never saw:

* those observations are genuinely out of sample, so the whole network can be
  used and there is no circularity;
* nothing is held out, so the training network equals the operational one and
  the bias towards long radii disappears;
* the cross-block carries the influence of every radius, observed location or
  not, into the k block where the score is taken, so there are no blind
  parameters;
* and no future data is needed: at cycle k, both k-1 and k are in the past.

The radius here controls two things at once, which is a feature of the
augmented formulation rather than an accident: how far a component may reach in
space, and how much of the previous time it may use as predecessors.

What has been measured, and what has not
----------------------------------------
Over twelve independent pairs of cycles at n = 40, N = 20, half the network
observed, the criterion and the truth are minimized at the same radius, and
they rise and fall together across the whole range rather than agreeing only at
the minimum. The criterion varies by a factor of 1.5 to 2.2 across the radius
range, against roughly 5% for the held-out cross-validation it replaces.

Not yet measured: behaviour over a long filter run rather than isolated pairs,
one radius per component rather than a uniform one, and sensitivity to the
choice of temporal predecessors, which is a design decision taken here and not
derived from anything.
"""
from __future__ import annotations

import numpy as np

from .taper import predecessors


# ----------------------------------------------------------------------
def ridge_solve(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression with the penalty scaled by the trace of X'X.

    An absolute constant on the diagonal does not regularize anything when the
    magnitude of X'X depends on the variance of the data. In Lorenz-96 the
    entries run to a few hundred per predecessor, so a fixed alpha of 0.01 is
    orders of magnitude below the diagonal and the augmented system becomes
    numerically singular where the predecessor count approaches the ensemble
    size. Scaling by the trace makes alpha a fraction of the typical variance
    and the estimator invariant to the scale of the state.
    """
    p = X.shape[1]
    G = X.T @ X
    G.flat[:: p + 1] += alpha * (np.trace(G) / p + 1e-12)
    return np.linalg.solve(G, X.T @ y)


def augmented_predecessors(i: int, n: int, radius: int,
                           temporal: bool = True) -> np.ndarray:
    """Predecessor set of component ``i`` of the augmented state.

    Components are ordered ``[x_{k-1}(0..n-1), x_k(0..n-1)]``. Within a time
    level the predecessors are the spatial neighbours with a lower index, as in
    the standard modified Cholesky construction. A component of the k block may
    additionally take the neighbourhood of the same site at k-1, which is what
    makes the cross-block non-zero and therefore what carries the influence of
    every radius into the block where the criterion is evaluated.

    The temporal rule is a design choice, not a derivation: a different one
    would give a different estimator, and the sensitivity to it has not been
    measured.
    """
    t, j = divmod(i, n)
    r = int(radius)
    if r <= 0:
        return np.empty(0, dtype=int)

    idx = [t * n + q for q in predecessors(n, j, r).tolist()]
    if t == 1 and temporal:
        idx += (np.unique(np.arange(j - r, j + r + 1) % n)).tolist()
    return np.array(sorted({q for q in idx if q < i}), dtype=int)


def augmented_precision(DZ: np.ndarray, n: int, r_field, alpha: float = 0.15,
                        r_max: int = 12, temporal: bool = True) -> np.ndarray:
    """B^{-1} of the augmented state from the modified Cholesky decomposition.

    ``DZ`` is the (2n, N) matrix of ensemble deviations of the joint vector and
    ``r_field`` is one radius per augmented component, so a spatially varying
    radius costs nothing extra here.
    """
    m = DZ.shape[0]
    r_field = np.asarray(r_field, dtype=float)
    if r_field.ndim == 0:
        r_field = np.full(m, float(r_field))
    r_int = np.clip(np.floor(r_field).astype(int), 0, r_max)

    L = np.eye(m)
    d = np.empty(m)
    for i in range(m):
        idx = augmented_predecessors(i, n, r_int[i], temporal)
        y = DZ[i, :]
        if idx.size == 0:
            v = float(np.var(y))
            d[i] = 1.0 / v if v > 0 else 0.0
            continue
        Xp = DZ[idx, :].T
        beta = ridge_solve(Xp, y, alpha)
        resid = y - Xp @ beta
        v = float(np.var(resid))
        d[i] = 1.0 / v if v > 0 else 0.0
        L[i, idx] = -beta
    return L.T @ (d[:, None] * L)


# ----------------------------------------------------------------------
class AugmentedSmoother:
    """One smoothing step over the pair (k-1, k).

    Everything that does not depend on the radius is built once: the joint
    ensemble, its mean and deviations, the observation operator of k-1 and the
    innovation. A candidate radius then rebuilds only the precision matrix and
    solves one (2n x 2n) system, with no model integration.
    """

    def __init__(self, Xb_prev, Xb_now, obs_idx_prev, y_prev, r_inv_prev,
                 obs_idx_now, y_now, r_inv_now, alpha=0.15, r_max=12,
                 temporal=True):
        self.n, self.N = Xb_prev.shape
        self.alpha, self.r_max, self.temporal = alpha, r_max, temporal

        Z = np.vstack([np.asarray(Xb_prev, float), np.asarray(Xb_now, float)])
        self.zb = Z.mean(axis=1)
        self.DZ = Z - self.zb[:, None]

        self.idx_prev = np.asarray(obs_idx_prev, int)
        self.y_prev = np.asarray(y_prev, float)
        self.ri_prev = np.asarray(r_inv_prev, float)
        self.idx_now = np.asarray(obs_idx_now, int)
        self.y_now = np.asarray(y_now, float)
        self.ri_now = np.asarray(r_inv_now, float)

        # H selects observed sites of the k-1 block only.
        H = np.zeros((self.idx_prev.size, 2 * self.n))
        H[np.arange(self.idx_prev.size), self.idx_prev] = 1.0
        self.H = H
        self.HtRinvH = H.T @ (self.ri_prev[:, None] * H)
        self.rhs = H.T @ (self.ri_prev * (self.y_prev - self.zb[self.idx_prev]))

    # ------------------------------------------------------------------
    def analyse(self, r_field):
        """Smoothed mean of the augmented state for a candidate radius."""
        Binv = augmented_precision(self.DZ, self.n, r_field, self.alpha,
                                   self.r_max, self.temporal)
        return self.zb + np.linalg.solve(Binv + self.HtRinvH, self.rhs)

    def criterion(self, r_field) -> float:
        """J(r): the k block scored against the observations of k.

        Admissible: the observations of k were not assimilated, no reference
        trajectory is used, and nothing is withheld from the network.
        """
        za = self.analyse(r_field)
        resid = self.y_now - za[self.n:][self.idx_now]
        return float(resid @ (self.ri_now * resid)) / resid.size

    def smoothed_prev(self, r_field) -> np.ndarray:
        """The k-1 block: the quantity the smoother is actually delivering."""
        return self.analyse(r_field)[:self.n]

    def truth_error(self, r_field, x_true_prev) -> float:
        """Reference only. Never used to select anything."""
        return float(np.sqrt(np.mean(
            (self.smoothed_prev(r_field) - np.asarray(x_true_prev)) ** 2)))


# ----------------------------------------------------------------------
def select_radius(smoother, optimizer="fpa", param=None, lo=1.0, hi=12.0,
                  budget=120, seed=0, x0=None, opt_params=None):
    """Minimize the criterion over the admissible box.

    ``param`` maps a parameter vector to one radius per augmented component;
    with ``None`` a single radius is used for the whole augmented state. The
    objective is integer-valued in the radius, so candidates that floor to the
    same predecessor sets are cached and cost one evaluation, not several.
    """
    from .metaheuristics import get_optimizer
    from .objective import BudgetExhausted

    m = 2 * smoother.n
    K = 1 if param is None else param.K
    expand = (lambda th: np.full(m, float(np.atleast_1d(th)[0]))) \
        if param is None else param.expand

    state = dict(n_evals=0, n_calls=0, n_cache_hits=0, stalled=False, cache={})
    trace = []

    class Obj:
        # make_info reads these off the objective, so they are attributes and
        # not just entries in the closure's dict.
        n_evals = 0
        n_calls = 0
        n_cache_hits = 0
        stalled = False

        def raw(self, th):
            return smoother.criterion(expand(th))

        def __call__(self, th):
            state["n_calls"] += 1
            if state["n_calls"] > budget * 25:
                state["stalled"] = True
                raise BudgetExhausted("call limit")
            key = tuple(np.floor(np.atleast_1d(th)).astype(int))
            if key in state["cache"]:
                state["n_cache_hits"] += 1
                self._sync()
                return state["cache"][key]
            if state["n_evals"] >= budget:
                raise BudgetExhausted("budget")
            v = self.raw(th)
            state["cache"][key] = v
            state["n_evals"] += 1
            trace.append(v)
            self._sync()
            return v

        def _sync(self):
            for k in ("n_evals", "n_calls", "n_cache_hits", "stalled"):
                setattr(self, k, state[k])

        def remaining(self):
            if state["stalled"]:
                return 0
            return max(0, budget - state["n_evals"])

        def best_so_far(self):
            return min(state["cache"].values()) if state["cache"] else np.nan

    obj = Obj()
    theta, info = get_optimizer(optimizer)(
        obj, np.full(K, lo), np.full(K, hi), budget,
        np.random.default_rng(seed), x0=x0, **(opt_params or {}))

    info.update(n_evals=state["n_evals"], n_calls=state["n_calls"],
                n_cache_hits=state["n_cache_hits"], stalled=state["stalled"],
                trace=list(np.minimum.accumulate(trace)) if trace else [])
    return theta, expand(theta), info
