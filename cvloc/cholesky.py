# -*- coding: utf-8 -*-
"""
The modified Cholesky parameterization: the radius as a predecessor set.

In the tapered formulation the radius enters through a smooth kernel, and the
objective inherits that smoothness: the landscape is flat towards short radii
and rises steeply towards long ones, and on one free parameter almost anything
finds the minimum. That is why the comparison between candidate searches in the
tapered case is uninformative, and it is worth being explicit about it rather
than presenting a tie as a result.

Here the radius determines which components enter a regression,

    x_[i] regressed on { j < i : d(i, j) <= r_i },
    B^{-1}(r) = L(r)' D(r)^{-1} L(r)

so the estimator changes only when the radius crosses an integer. The objective
is piecewise constant with jumps, the flat regions are genuine plateaus rather
than a noise floor, and a search that merely walks downhill has something to
fail at. This is the setting in which the choice of metaheuristic can actually
be discriminated, which is the point of EXP-10.

The regression uses the closed-form ridge solution; with fit_intercept=False
this is numerically the estimator of ``pyteda.analysis.AnalysisEnKFModifiedCholesky``.
"""
from __future__ import annotations

import numpy as np

from .taper import predecessors


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """(X'X + alpha I)^{-1} X'y, with X of shape (N, p)."""
    p = X.shape[1]
    G = X.T @ X
    G.flat[:: p + 1] += alpha
    return np.linalg.solve(G, X.T @ y)


class CholeskyPrecision:
    """B^{-1}(r) from the modified Cholesky decomposition.

    The predecessor sets are cached by (i, integer radius), so a search that
    revisits a radius pays for the regressions only once per cycle.
    """

    def __init__(self, n: int, ridge_alpha: float = 0.01, r_max: int = 12):
        self.n = int(n)
        self.ridge_alpha = float(ridge_alpha)
        self.r_max = int(r_max)
        self._pre = {}
        self.DX = None
        self._cache = {}

    def build(self, DX: np.ndarray):
        """Attach the deviation matrix (n, N) and clear the per-cycle cache."""
        self.DX = np.asarray(DX, dtype=float)
        self._cache = {}
        return self

    def _predecessors(self, i: int, r: int) -> np.ndarray:
        key = (i, int(r))
        if key not in self._pre:
            self._pre[key] = predecessors(self.n, i, int(r))
        return self._pre[key]

    def precision(self, r) -> np.ndarray:
        """B^{-1} for a radius field, rounded to integers.

        ``r`` may be a scalar or one value per component. The result is cached
        on the integer radius vector, so the piecewise-constant structure of
        the landscape is reflected in the evaluation count: candidates that
        round to the same predecessor sets cost nothing extra.
        """
        if self.DX is None:
            raise RuntimeError("call build(DX) before precision(r)")
        r_arr = np.asarray(r, dtype=float)
        if r_arr.ndim == 0:
            r_arr = np.full(self.n, float(r_arr))
        r_int = np.clip(np.floor(r_arr).astype(int), 0, self.r_max)

        key = tuple(r_int.tolist())
        if key in self._cache:
            return self._cache[key]

        DX = self.DX
        n = self.n
        L = np.eye(n)
        d = np.empty(n)
        v0 = float(np.var(DX[0, :]))
        d[0] = 1.0 / v0 if v0 > 0 else 0.0
        for i in range(1, n):
            idx = self._predecessors(i, r_int[i])
            y = DX[i, :]
            if idx.size == 0:
                v = float(np.var(y))
                d[i] = 1.0 / v if v > 0 else 0.0
                continue
            Xp = DX[idx, :].T
            beta = _ridge(Xp, y, self.ridge_alpha)
            resid = y - Xp @ beta
            v = float(np.var(resid))
            d[i] = 1.0 / v if v > 0 else 0.0
            L[i, idx] = -beta
        Binv = L.T @ (d[:, None] * L)
        self._cache[key] = Binv
        return Binv


class CholeskySpace:
    """The Cholesky analogue of :class:`cvloc.ensemble_space.EnsembleSpace`.

    Same interface as far as the objective is concerned: give it a radius and
    an observation mask, get back an analysis mean. What differs is that the
    radius rebuilds a precision matrix instead of re-weighting a diagonal, so
    an evaluation is more expensive here and the budget is the binding
    constraint rather than a formality.
    """

    def __init__(self, cycle, ridge_alpha: float = 0.01, r_max: int = 12):
        self.cycle = cycle
        self.n, self.N = cycle.Xb.shape
        self.xb = cycle.Xb.mean(axis=1)
        self.DX = cycle.Xb - self.xb[:, None]
        self.obs_idx = np.asarray(cycle.obs_idx, dtype=int)
        self.y = np.asarray(cycle.y, dtype=float)
        self.r_inv = np.asarray(cycle.r_inv, dtype=float)
        self.family = CholeskyPrecision(self.n, ridge_alpha, r_max).build(self.DX)

        H = np.zeros((self.obs_idx.size, self.n))
        H[np.arange(self.obs_idx.size), self.obs_idx] = 1.0
        self.H = H
        self.innov = self.y - self.xb[self.obs_idx]

    def analysis_mean(self, r, mask=None) -> np.ndarray:
        Binv = self.family.precision(r)
        ri = self.r_inv if mask is None else self.r_inv * np.asarray(mask, float)
        HtRinvH = self.H.T @ (ri[:, None] * self.H)
        rhs = self.H.T @ (ri * self.innov)
        return self.xb + np.linalg.solve(Binv + HtRinvH, rhs)

    def analysis_ensemble(self, r, mask=None, rng=None):
        """Stochastic analysis ensemble, for the multi-cycle setting."""
        rng = np.random.default_rng(0) if rng is None else rng
        Binv = self.family.precision(r)
        ri = self.r_inv if mask is None else self.r_inv * np.asarray(mask, float)
        noise = rng.standard_normal((self.obs_idx.size, self.N)) / np.sqrt(self.r_inv)[:, None]
        D = (self.y[:, None] + noise) - self.H @ self.cycle.Xb
        HtRinvH = self.H.T @ (ri[:, None] * self.H)
        Z = np.linalg.solve(Binv + HtRinvH, self.H.T @ (ri[:, None] * D))
        Xa = self.cycle.Xb + Z
        return Xa.mean(axis=1), Xa

    def rmse(self, xa) -> float:
        return float(np.sqrt(np.mean((xa - self.cycle.x_true) ** 2)))

    def truth_error(self, r) -> float:
        return self.rmse(self.analysis_mean(r))
