# -*- coding: utf-8 -*-
"""
Parameterizations of the radius field.

The search always happens in a box in R^K; what changes between experiments is
how those K numbers are spread over the n analysis points. Separating the two
is what lets EXP-07 vary K while holding the search, the objective and the
budget fixed, so that the effect of the parameterization can be told apart
from the effect of the search.

Four maps are provided:

``Uniform``      K = 1, one radius for the whole domain. The configuration the
                 draft reports as the one that works.
``Blocks``       K contiguous blocks of grid points, K free between 1 and n.
                 The sweep of EXP-07.
``PerVariable``  K = n, one radius per state variable. The extreme of the
                 sweep, and the one the draft reports as degrading.
``Clusters``     one radius per group of grid points, where the groups come
                 from ensemble-derived features rather than from position.
                 The parsimonious alternative the discussion argues for.
"""
from __future__ import annotations

import numpy as np


class Parameterization:
    """Map from a parameter vector theta in R^K to a radius per grid point."""

    name = "base"

    def __init__(self, n: int, K: int):
        self.n = int(n)
        self.K = int(K)

    def expand(self, theta) -> np.ndarray:
        raise NotImplementedError

    def label(self) -> str:
        return f"{self.name}(K={self.K})"

    def __repr__(self) -> str:
        return f"<{self.label()} n={self.n}>"


class Uniform(Parameterization):
    """One radius for the whole domain."""

    name = "uniform"

    def __init__(self, n: int):
        super().__init__(n, 1)

    def expand(self, theta) -> np.ndarray:
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        return np.full(self.n, float(theta[0]))


class Blocks(Parameterization):
    """K contiguous blocks of grid points, one radius each.

    Blocks are as equal in size as the grid allows. With K = 1 this is
    ``Uniform`` and with K = n it is ``PerVariable``, so a single class covers
    the whole sweep and no result can be attributed to a change of code path.
    """

    name = "blocks"

    def __init__(self, n: int, K: int):
        super().__init__(n, K)
        if not 1 <= self.K <= self.n:
            raise ValueError(f"K must be between 1 and {n}, got {K}")
        self.assign = (np.arange(self.n) * self.K) // self.n

    def expand(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return theta[self.assign]


class PerVariable(Blocks):
    """One free radius per state variable."""

    name = "per-variable"

    def __init__(self, n: int):
        super().__init__(n, n)


class Clusters(Parameterization):
    """One radius per group, with groups given by a label vector.

    The labels normally come from :mod:`cvloc.features`, that is from the
    background variance and the correlation decay length of the ensemble, not
    from the position on the grid. Two distant grid points that behave alike
    then share a radius, which is the point: the number of parameters stays
    small while the radius is still allowed to vary in space.
    """

    name = "clusters"

    def __init__(self, labels):
        labels = np.asarray(labels, dtype=int)
        uniq = np.unique(labels)
        remap = {u: k for k, u in enumerate(uniq)}
        self.assign = np.array([remap[v] for v in labels], dtype=int)
        super().__init__(labels.size, uniq.size)

    def expand(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return theta[self.assign]


# ----------------------------------------------------------------------
def make(kind: str, n: int, K: int = 1, labels=None) -> Parameterization:
    if kind == "uniform":
        return Uniform(n)
    if kind == "blocks":
        return Blocks(n, K)
    if kind == "per-variable":
        return PerVariable(n)
    if kind == "clusters":
        if labels is None:
            raise ValueError("the clusters parameterization needs labels")
        return Clusters(labels)
    raise ValueError(f"unknown parameterization '{kind}'")


# ----------------------------------------------------------------------
# Admissible boxes
# ----------------------------------------------------------------------
#
# Yang's second point: without bounds, or with bounds far wider than the useful
# range, most of the population spends its evaluations where the objective is
# flat. The two boxes below make that testable rather than assumed. The wide
# box is the naive default; the informed box is what an educated guess about
# the correlation length of the model would give.
# The upper end matters more than it looks. With a fully observed network the
# optimal radius sits below two, and a box ending at eight is generous. With
# half the network observed it runs to ten, and with a network that moves every
# cycle the cycled optimum is near eight; a box ending at eight then pins the
# search against its own boundary and the resulting penalty measures the box,
# not the criterion. Every box below therefore contains the optimum of every
# regime in the suite, and `wide` is the default. `informed` and `tight` exist
# so that EXP-07 can measure what tightening the box buys.
BOXES = {
    "wide": (0.2, 20.0),
    "informed": (0.5, 12.0),
    "tight": (1.0, 6.0),
}


def box(name: str, K: int):
    """Return ``(lo, hi)`` arrays of length K for a named admissible box."""
    if name not in BOXES:
        raise ValueError(f"unknown box '{name}'. Available: {sorted(BOXES)}")
    lo, hi = BOXES[name]
    return np.full(K, float(lo)), np.full(K, float(hi))
