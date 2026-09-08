# -*- coding: utf-8 -*-
"""
Ensemble-derived features and the grouping built from them.

The discussion of the draft argues that a spatially varying radius can only pay
where grid points are not statistically exchangeable, and that the parameters
must still be few. Both together point at grouping variables by quantities the
ensemble already provides, and estimating one radius per group.

Two features are used.

``sigma2``  the background variance at the grid point. Where the ensemble is
            confident the analysis leans on the background and a shorter radius
            costs little; where it is not, remote information is worth more.
``decay``   the correlation decay length: the lag at which the ensemble
            correlation with the neighbouring points first falls below 1/e,
            interpolated linearly between lags. This is a direct sample estimate
            of the length scale the radius is supposed to respect.

Both are computable from the forecast ensemble alone, so a grouping built from
them is admissible in the same sense as the criterion itself.
"""
from __future__ import annotations

import numpy as np


def background_variance(Xb: np.ndarray) -> np.ndarray:
    """Per grid point sample variance of the ensemble."""
    return np.var(np.asarray(Xb, dtype=float), axis=1, ddof=1)


def correlation_decay(Xb: np.ndarray, max_lag: int = None,
                      threshold: float = None) -> np.ndarray:
    """Per grid point correlation decay length, in grid units.

    For each point j the ensemble correlation with j+l and j-l is averaged over
    the two directions, and the length is the lag at which that average first
    drops below ``threshold`` (1/e by default), with linear interpolation
    between the bracketing lags. Points whose correlation never drops within
    ``max_lag`` get ``max_lag``.
    """
    X = np.asarray(Xb, dtype=float)
    n, N = X.shape
    if max_lag is None:
        max_lag = max(2, n // 4)
    if threshold is None:
        threshold = float(np.exp(-1.0))

    A = X - X.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.sum(A * A, axis=1))
    sd[sd == 0.0] = 1e-12
    A = A / sd[:, None]

    lags = np.arange(1, max_lag + 1)
    corr = np.empty((n, max_lag))
    for k, l in enumerate(lags):
        fwd = np.einsum("ij,ij->i", A, np.roll(A, -l, axis=0))
        bwd = np.einsum("ij,ij->i", A, np.roll(A, l, axis=0))
        corr[:, k] = 0.5 * (np.abs(fwd) + np.abs(bwd))

    out = np.full(n, float(max_lag))
    for j in range(n):
        c = corr[j]
        below = np.flatnonzero(c < threshold)
        if below.size == 0:
            continue
        i = int(below[0])
        if i == 0:
            c0, c1 = 1.0, c[0]
            l0, l1 = 0.0, 1.0
        else:
            c0, c1 = c[i - 1], c[i]
            l0, l1 = float(lags[i - 1]), float(lags[i])
        denom = (c0 - c1)
        frac = 0.0 if denom <= 0 else (c0 - threshold) / denom
        out[j] = l0 + frac * (l1 - l0)
    return out


def feature_matrix(Xb: np.ndarray) -> np.ndarray:
    """(n, 2) matrix of [background variance, decay length]."""
    return np.column_stack([background_variance(Xb), correlation_decay(Xb)])


def standardize(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    mu, sd = F.mean(axis=0), F.std(axis=0)
    sd[sd == 0.0] = 1.0
    return (F - mu) / sd


def cluster(Xb: np.ndarray, K=None, K_range=(2, 8), seed: int = 0):
    """Group grid points by ensemble features.

    With ``K`` given, runs k-means for that many groups. With ``K`` left as
    ``None``, the number of groups is chosen by the silhouette score over
    ``K_range``, which is the procedure the draft's Figure 1(e) refers to.

    Returns ``(labels, features, info)``.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    F = feature_matrix(Xb)
    Z = standardize(F)

    if K is not None:
        km = KMeans(n_clusters=int(K), n_init=10, random_state=seed).fit(Z)
        return km.labels_, F, dict(K=int(K), selected_by="given",
                                   silhouette=float("nan"))

    lo, hi = K_range
    hi = min(hi, Z.shape[0] - 1)
    best = (None, -np.inf, None)
    scores = {}
    for k in range(max(2, lo), hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        if len(np.unique(km.labels_)) < 2:
            continue
        s = float(silhouette_score(Z, km.labels_))
        scores[k] = s
        if s > best[1]:
            best = (km.labels_, s, k)

    labels, score, k = best
    if labels is None:  # degenerate ensemble, fall back to one group
        labels = np.zeros(Z.shape[0], dtype=int)
        k, score = 1, float("nan")
    return labels, F, dict(K=int(k), selected_by="silhouette",
                           silhouette=float(score), scores=scores)
