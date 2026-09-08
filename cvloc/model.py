# -*- coding: utf-8 -*-
"""
Lorenz 96 with a vectorized RK4 integrator and optionally heterogeneous
forcing.

Two things are added on top of ``pyteda.models.Lorenz96``.

Speed. The upstream right-hand side is a Python list comprehension integrated
with ``odeint``. The suite evaluates the objective tens of thousands of times
and generates long climatological runs, so the right-hand side is rewritten
with ``np.roll`` and integrated with the fixed-step RK4 the draft specifies
(dt = 0.01). The trajectories agree with the upstream ones to integration
accuracy.

Heterogeneity. ``F`` may be an array of length ``n``. A domain whose forcing
varies in space is no longer statistically exchangeable across grid points,
which is the setting in which a spatially varying radius has something to gain.
That is the whole point of EXP-08, and it cannot be expressed with the scalar
forcing of the upstream class.
"""
from __future__ import annotations

import numpy as np

from pyteda.models import Lorenz96


class Lorenz96CV(Lorenz96):
    """Lorenz 96 with RK4 and per-component forcing.

    Parameters
    ----------
    n : int
        Number of grid points.
    F : float or array of shape (n,)
        Forcing. A scalar reproduces the homogeneous model; an array makes the
        grid points non-exchangeable.
    dt : float
        Integration step of the RK4 scheme.
    """

    def __init__(self, n: int = 40, F=8.0, dt: float = 0.01):
        F_arr = np.asarray(F, dtype=float)
        if F_arr.ndim == 0:
            super().__init__(n=n, F=float(F_arr))
            self.F_field = np.full(n, float(F_arr))
            self.heterogeneous = False
        else:
            if F_arr.shape != (n,):
                raise ValueError(f"F must be scalar or shape ({n},), got {F_arr.shape}")
            super().__init__(n=n, F=float(F_arr.mean()))
            self.F_field = F_arr.copy()
            self.heterogeneous = True
        self.dt = float(dt)

    # ------------------------------------------------------------------
    def rhs(self, x: np.ndarray) -> np.ndarray:
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + self.F_field

    def _rk4_step(self, x: np.ndarray, h: float) -> np.ndarray:
        k1 = self.rhs(x)
        k2 = self.rhs(x + 0.5 * h * k1)
        k3 = self.rhs(x + 0.5 * h * k2)
        k4 = self.rhs(x + h * k3)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def propagate(self, x0: np.ndarray, T, just_final_state: bool = True):
        """Integrate from ``T[0]`` to ``T[-1]``.

        The signature is the one pyteda's Simulation expects, so this class is
        a drop-in replacement inside a Scenario.
        """
        T = np.atleast_1d(np.asarray(T, dtype=float))
        t0, t1 = float(T[0]), float(T[-1])
        span = t1 - t0
        if span <= 0:
            return np.asarray(x0, dtype=float).copy()
        n_steps = max(1, int(np.ceil(span / self.dt)))
        h = span / n_steps
        x = np.asarray(x0, dtype=float).copy()
        if just_final_state:
            for _ in range(n_steps):
                x = self._rk4_step(x, h)
            return x
        traj = np.empty((n_steps + 1, x.size))
        traj[0] = x
        for k in range(n_steps):
            x = self._rk4_step(x, h)
            traj[k + 1] = x
        return traj

    def get_initial_condition(self, seed: int = 10, T=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x0 = self.F_field.mean() * np.ones(self.n)
        x0 += 0.01 * rng.standard_normal(self.n)
        return self.propagate(x0, np.array([0.0, 20.0]))

    # ------------------------------------------------------------------
    def free_run(self, x0: np.ndarray, spacing: float, n_snapshots: int) -> np.ndarray:
        """Snapshots of a free run separated by ``spacing`` time units.

        Returns an array of shape ``(n, n_snapshots)``. With a spacing of order
        one the snapshots are effectively independent draws from the model
        climatology, which is the background ensemble the draft uses.
        """
        out = np.empty((self.n, n_snapshots))
        x = np.asarray(x0, dtype=float).copy()
        T = np.array([0.0, float(spacing)])
        for k in range(n_snapshots):
            x = self.propagate(x, T)
            out[:, k] = x
        return out


# ----------------------------------------------------------------------
# Heterogeneity profiles
# ----------------------------------------------------------------------
def forcing_profile(n: int, kind: str = "uniform", F_lo: float = 5.0,
                    F_hi: float = 12.0, n_regions: int = 4) -> np.ndarray:
    """Forcing field over the ring.

    ``uniform``  constant F_hi/F_lo midpoint, the homogeneous control.
    ``blocks``   ``n_regions`` contiguous regions alternating between F_lo and
                 F_hi, so the correlation length changes abruptly at region
                 boundaries.
    ``smooth``   a single sinusoid between F_lo and F_hi, so the correlation
                 length varies continuously and no radius is right everywhere.
    """
    j = np.arange(n)
    if kind == "uniform":
        return np.full(n, 0.5 * (F_lo + F_hi))
    if kind == "blocks":
        block = (j * n_regions) // n
        return np.where(block % 2 == 0, F_lo, F_hi).astype(float)
    if kind == "smooth":
        mid, amp = 0.5 * (F_lo + F_hi), 0.5 * (F_hi - F_lo)
        return mid + amp * np.sin(2.0 * np.pi * j / n)
    raise ValueError(f"unknown forcing profile '{kind}'")


def observation_network(n: int, kind: str = "full", rng=None,
                        dense_frac: float = 0.9, sparse_frac: float = 0.3,
                        n_regions: int = 4) -> np.ndarray:
    """Indices of the observed grid points.

    ``full``     every point observed, the configuration of the draft.
    ``uniform``  a random subset of uniform density.
    ``patchy``   alternating dense and sparse halves of the ring. The optimal
                 radius should be larger where observations are scarce, which
                 is the effect a spatially varying radius is supposed to
                 capture.
    """
    rng = np.random.default_rng() if rng is None else rng
    j = np.arange(n)
    if kind == "full":
        return j
    if kind == "uniform":
        m = max(1, int(round(dense_frac * n)))
        return np.sort(rng.choice(n, size=m, replace=False))
    if kind == "patchy":
        block = (j * n_regions) // n
        frac = np.where(block % 2 == 0, dense_frac, sparse_frac)
        keep = rng.random(n) < frac
        if not keep.any():
            keep[0] = True
        return j[keep]
    raise ValueError(f"unknown observation network '{kind}'")
