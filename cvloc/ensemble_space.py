# -*- coding: utf-8 -*-
"""
R-localized analysis in ensemble space.

Write ``x = xb + X w`` with ``X`` the background anomalies scaled by
``(N-1)^{-1/2}``. The background term becomes ``w'w`` because the anomalies are
whitened by construction, so localization cannot act multiplicatively on B in
this representation and is applied to the observation term instead:

    J_r(w) = 1/2 w'w + 1/2 (d - Y w)' Rtilde^{-1}(r) (d - Y w)

    Rtilde^{-1}(r) = R^{-1} o diag( rho(d_i / r_j) )

The whole scheme is affordable because **Y = H X and d = y - H xb do not depend
on r**. They are formed once per cycle; changing a candidate radius only
re-weights a diagonal and requires an N x N solve per analysis point, with no
model integration anywhere.

The solves for all n analysis points are batched into one ``np.linalg.solve``
call on a stack of (N, N) systems, which is what makes a full objective
evaluation cost a few hundred microseconds at n = 40, N = 20 and turns a
population search over thousands of candidates into a matter of seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .taper import obs_distance, taper_weights


@dataclass
class Cycle:
    """One assimilation cycle: everything frozen except the radius.

    A cycle is the unit of every experiment in this suite except EXP-09. It
    carries the truth so that the truth-based optimum can be computed as a
    reference, but nothing the *method* touches ever reads it.
    """
    Xb: np.ndarray            # (n, N) background ensemble
    x_true: np.ndarray        # (n,)   true state, reference only
    obs_idx: np.ndarray       # (p,)   observed grid points
    y: np.ndarray             # (p,)   observations
    r_inv: np.ndarray         # (p,)   inverse observation error variances
    model: object = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.Xb.shape[0]

    @property
    def N(self) -> int:
        return self.Xb.shape[1]

    @property
    def p(self) -> int:
        return self.y.size

    def background_rmse(self) -> float:
        return float(np.sqrt(np.mean((self.Xb.mean(axis=1) - self.x_true) ** 2)))


class EnsembleSpace:
    """Precomputed ensemble-space quantities for one cycle.

    Everything that does not depend on the radius is built here, once. The
    methods below then take a radius (scalar, or one value per analysis point)
    and an optional observation mask, and return the analysis.
    """

    def __init__(self, cycle: Cycle):
        self.cycle = cycle
        self.n, self.N = cycle.Xb.shape
        self.xb = cycle.Xb.mean(axis=1)
        self.X = (cycle.Xb - self.xb[:, None]) / np.sqrt(self.N - 1.0)
        self.obs_idx = np.asarray(cycle.obs_idx, dtype=int)
        self.Y = self.X[self.obs_idx, :]                    # (p, N)
        self.d = np.asarray(cycle.y, dtype=float) - self.xb[self.obs_idx]
        self.r_inv = np.asarray(cycle.r_inv, dtype=float)
        self.dist = obs_distance(self.n, self.obs_idx)      # (n, p)
        self.p = self.obs_idx.size
        self._eye = np.eye(self.N)

    # ------------------------------------------------------------------
    def weights(self, r, mask=None) -> np.ndarray:
        """(n, p) matrix ``rho(d_ij / r_j) / sigma_i^2``, zeroed where masked.

        Withholding an observation is exactly setting its taper weight to zero,
        so switching folds costs nothing: no factorization is reused across
        candidates in the first place.

        Passing ``r=None`` means no localization at all: the taper is exactly
        one everywhere. That is not the same as a very large radius, since the
        Gaspari-Cohn function equals one only at zero distance and approaches
        it from below; the unlocalized filter deserves an exact code path
        rather than an approximation that a reader would have to trust.
        """
        if r is None:
            W = np.broadcast_to(self.r_inv[None, :], self.dist.shape).copy()
        else:
            W = taper_weights(self.dist, r) * self.r_inv[None, :]
        if mask is not None:
            W = W * np.asarray(mask, dtype=float)[None, :]
        return W

    def _solve(self, W: np.ndarray) -> np.ndarray:
        """Batched local solves. Returns the (n, N) matrix of local weights."""
        # A[j] = I + Y' diag(W[j]) Y ,  b[j] = Y' diag(W[j]) d
        A = np.einsum("ik,ji,il->jkl", self.Y, W, self.Y, optimize=True)
        A += self._eye
        b = np.einsum("ik,ji,i->jk", self.Y, W, self.d, optimize=True)
        # numpy >= 2 requires the right-hand side of a batched solve to carry
        # an explicit trailing axis, otherwise (n, N) is read as a single
        # matrix rather than a stack of n vectors.
        return np.linalg.solve(A, b[..., None])[..., 0]

    # ------------------------------------------------------------------
    def analysis_mean(self, r, mask=None) -> np.ndarray:
        """Analysis mean for a candidate radius. No model integration."""
        w = self._solve(self.weights(r, mask))
        return self.xb + np.einsum("jk,jk->j", self.X, w)

    def analysis_ensemble(self, r, mask=None):
        """Analysis mean and ensemble, via the local transform (LETKF).

        Used at the end of a cycle, and by the multi-cycle experiment, where
        the analysis ensemble has to be propagated. Costs one batched
        eigendecomposition, so it is kept out of the objective.
        """
        W = self.weights(r, mask)
        A = np.einsum("ik,ji,il->jkl", self.Y, W, self.Y, optimize=True)
        A += self._eye
        b = np.einsum("ik,ji,i->jk", self.Y, W, self.d, optimize=True)
        w = np.linalg.solve(A, b[..., None])[..., 0]
        xa = self.xb + np.einsum("jk,jk->j", self.X, w)

        s, U = np.linalg.eigh(A)                             # (n, N), (n, N, N)
        s = np.maximum(s, 1e-12)
        T = np.einsum("jke,je,jle->jkl", U, 1.0 / np.sqrt(s), U, optimize=True)
        DXa = np.sqrt(self.N - 1.0) * np.einsum("jk,jkl->jl", self.X, T,
                                                optimize=True)
        return xa, xa[:, None] + DXa

    # ------------------------------------------------------------------
    def rmse(self, xa: np.ndarray) -> float:
        return float(np.sqrt(np.mean((xa - self.cycle.x_true) ** 2)))

    def truth_error(self, r) -> float:
        """RMSE against the truth, with the full observation set.

        This is the oracle criterion. It is not admissible as a method: it is
        the quantity the cross-validated criterion is trying to imitate without
        being allowed to look at it.
        """
        return self.rmse(self.analysis_mean(r))

    def sample_covariance(self) -> np.ndarray:
        return self.X @ self.X.T
