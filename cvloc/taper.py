# -*- coding: utf-8 -*-
"""
Gaspari-Cohn taper and the distance geometry of the Lorenz 96 ring.

The localization radius enters the analysis only through the weight

    rho(d_ij / r_j)

where ``d_ij`` is the distance from observation ``i`` to the analysis point
``j`` and ``r_j`` is the radius assigned to that point. Everything else in the
assimilation is independent of the radius, which is what makes a candidate
cheap to score.

Convention: ``r`` is the *half-width* ``c`` of Gaspari and Cohn (1999), so the
taper vanishes beyond ``2r``. This matches ``rho(d_i / r_j)`` as written in the
draft and is the convention used throughout the suite; anyone comparing radii
against a paper that uses the support length instead must divide by two.
"""
from __future__ import annotations

import numpy as np


def gaspari_cohn(z):
    """Fifth-order piecewise rational correlation function of Gaspari-Cohn.

    Parameters
    ----------
    z : array_like
        Normalized distance ``d / c``. The function is 1 at 0, about 0.208 at
        1, and identically 0 beyond 2.
    """
    z = np.abs(np.asarray(z, dtype=float))
    out = np.zeros_like(z)

    near = z <= 1.0
    far = (z > 1.0) & (z <= 2.0)

    a = z[near]
    out[near] = ((((-0.25 * a + 0.5) * a + 0.625) * a - 5.0 / 3.0) * a * a) + 1.0

    b = z[far]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[far] = (((((b / 12.0 - 0.5) * b + 0.625) * b + 5.0 / 3.0) * b - 5.0) * b
                    + 4.0 - 2.0 / (3.0 * b))
    return np.clip(out, 0.0, 1.0)


def cyclic_distance(n: int) -> np.ndarray:
    """(n, n) matrix of distances on a ring of n grid points."""
    i = np.arange(n)
    diff = np.abs(i[:, None] - i[None, :])
    return np.minimum(diff, n - diff).astype(float)


def obs_distance(n: int, obs_idx) -> np.ndarray:
    """(n, p) distances from every grid point j to every observed point i.

    Rows are analysis points, columns are observations, which is the layout the
    weight matrix needs.
    """
    obs_idx = np.asarray(obs_idx, dtype=int)
    j = np.arange(n)[:, None]
    diff = np.abs(j - obs_idx[None, :])
    return np.minimum(diff, n - diff).astype(float)


def taper_weights(dist: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Row-wise taper: ``W[j, i] = rho(dist[j, i] / r[j])``.

    ``r`` has one entry per analysis point, so a spatially varying radius costs
    nothing extra here.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 0:
        r = np.full(dist.shape[0], float(r))
    return gaspari_cohn(dist / np.maximum(r, 1e-12)[:, None])


def taper_matrix(n: int, r, combine: str = "mean") -> np.ndarray:
    """(n, n) Gaspari-Cohn taper of the background covariance.

    Only used to draw the covariance panels of the figures; the analysis itself
    localizes the observation term, not B. With a spatially varying radius the
    pair (i, j) needs a single number, and the two usual rules are the mean and
    the minimum of the two radii.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 0:
        r = np.full(n, float(r))
    d = cyclic_distance(n)
    if combine == "mean":
        rr = 0.5 * (r[:, None] + r[None, :])
    elif combine == "min":
        rr = np.minimum(r[:, None], r[None, :])
    else:
        raise ValueError(f"combine must be 'mean' or 'min', got '{combine}'")
    return gaspari_cohn(d / np.maximum(rr, 1e-12))


def predecessors(n: int, i: int, radius: float) -> np.ndarray:
    """Indices ``j < i`` within ``radius`` of ``i`` on the ring.

    This is the predecessor set of the modified Cholesky decomposition. Note
    that the radius acts here as an integer count of grid points, not as the
    half-width of a smooth kernel: the set changes only when the radius crosses
    an integer, which is exactly the discreteness that makes this landscape
    different from the tapered one.
    """
    r = int(np.floor(radius))
    if r <= 0:
        return np.empty(0, dtype=int)
    ngb = np.arange(i - r, i + r + 1) % n
    return np.unique(ngb[ngb < i])
