# -*- coding: utf-8 -*-
"""
Persistence of the objects a figure needs and a CSV cannot hold.

A metrics table answers "how well did it do". It cannot answer "what did the
estimator look like", which is the question every covariance panel, taper plot
and radius profile in the paper is really asking. Those need the ensemble
itself, and an ensemble does not fit in a row.

Two tiers, because the two questions have very different costs.

**Trajectory tier**, kept for every run. The analysis and background *means* at
every cycle, the truth, and every per-cycle metric: absolute and relative RMSE,
ensemble spread, CRPS, the radius in use and the raw estimate before smoothing,
the attained objective, the evaluation count, and how many observations there
were. Three vectors of n floats per cycle is a few hundred kilobytes for a
300-cycle run, so this can be kept for everything and is what any time series
in the paper is drawn from. Means are stored as float32: they are for plotting,
not for restarting a filter.

**Ensemble tier**, kept for a handful of cycles of a handful of methods. The
full forecast and analysis ensembles. This is what a covariance panel needs,
and it is two orders of magnitude larger per cycle, so it is deliberately
sparse.

What is stored in the ensemble tier, and why each item is not derivable from
the others:

``Xb``, ``Xa``   the forecast and analysis ensembles. Everything else in a
                 covariance figure is a function of these, so storing them
                 means a panel can be redrawn, restyled or recomputed with a
                 different taper without rerunning the assimilation.
``x_true``       the truth. Needed to draw the error, and to recompute the
                 oracle if the sweep is refined later.
``obs_idx``      which grid points were observed. With a moving network this
                 changes every cycle and cannot be reconstructed from a seed
                 without replaying the whole scenario.
``y``, ``r_inv`` the observations and their precisions.
``r``            the radius field actually used, and the raw estimate before
                 smoothing, so the two can be plotted against each other.

Everything goes into one compressed ``.npz`` per experiment. The arrays are
stacked along a leading cycle axis and an index CSV records what each slice
is, so a plot script can select without loading the whole file.

Size. A single cycle at n = 40, N = 20 is about 13 kB uncompressed. Storing
every cycle of every run in EXP-09 at paper scale would be some hundreds of
megabytes, which is why the default stores a handful of cycles per run rather
than all of them: ``store_states_at`` takes fractions of the run, so 0, 0.5
and 1 give the beginning, middle and end. The single-cycle experiments store
everything, since there is only one cycle each.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


class SnapshotWriter:
    """Collects ensembles and their context, then writes one npz plus an index.

    Usage:

        snaps = SnapshotWriter()
        snaps.add(tag="EXP-01/cycle3", Xb=..., Xa=..., x_true=..., ...)
        snaps.write(ctx.path("snapshots.npz"), ctx.path("snapshots_index.csv"))
    """

    def __init__(self):
        self.records = []

    def add(self, tag, Xb=None, Xa=None, x_true=None, obs_idx=None, y=None,
            r_inv=None, r=None, r_raw=None, **meta):
        """Add an ensemble-tier record: one cycle, with its full ensembles."""
        rec = dict(tag=str(tag), meta=dict(meta, tier="ensemble"))
        for name, arr in (("Xb", Xb), ("Xa", Xa), ("x_true", x_true),
                          ("obs_idx", obs_idx), ("y", y), ("r_inv", r_inv),
                          ("r", r), ("r_raw", r_raw)):
            if arr is not None:
                rec[name] = np.asarray(arr)
        self.records.append(rec)
        return self

    def add_trajectory(self, tag, xb_mean=None, xa_mean=None, x_true=None,
                       metrics=None, **meta):
        """Add a trajectory-tier record: the whole run, means and metrics only.

        ``xb_mean``, ``xa_mean`` and ``x_true`` are (n_cycles, n_state).
        ``metrics`` is a mapping of name to a length-n_cycles array, and each
        entry becomes its own key so a reader can pull one series without
        loading the states.
        """
        rec = dict(tag=str(tag), meta=dict(meta, tier="trajectory"))
        for name, arr in (("xb_mean", xb_mean), ("xa_mean", xa_mean),
                          ("x_true", x_true)):
            if arr is not None:
                rec[name] = np.asarray(arr, dtype=np.float32)
        for name, arr in (metrics or {}).items():
            arr = np.asarray(arr)
            if arr.size:
                rec[f"m_{name}"] = arr.astype(np.float32)
        self.records.append(rec)
        return self

    def __len__(self):
        return len(self.records)

    def write(self, npz_path, index_path=None, compress=True):
        """Write the archive. Returns the path, or None when empty.

        Arrays are stored one key per record rather than stacked, because with
        a moving observation network the number of observations differs between
        cycles and a stacked array would need padding. The index CSV carries
        the shapes so a reader knows what it is getting before it loads.
        """
        if not self.records:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)

        payload, rows = {}, []
        for i, rec in enumerate(self.records):
            row = dict(slot=i, tag=rec["tag"])
            for name, arr in rec.items():
                if name in ("tag", "meta"):
                    continue
                payload[f"{i:05d}/{name}"] = arr
                row[f"shape_{name}"] = "x".join(str(d) for d in arr.shape)
            row.update(rec["meta"])
            rows.append(row)

        (np.savez_compressed if compress else np.savez)(npz_path, **payload)
        if index_path:
            pd.DataFrame(rows).to_csv(index_path, index=False)
        return npz_path


def load_snapshots(npz_path, index_path=None):
    """Read an archive back. Returns ``(records, index)``.

    ``records`` is a list of dicts of arrays in slot order, so
    ``records[3]["Xb"]`` is the forecast ensemble of the fourth stored cycle.
    """
    z = np.load(npz_path)
    slots = sorted({int(k.split("/")[0]) for k in z.files})
    records = []
    for s in slots:
        prefix = f"{s:05d}/"
        records.append({k[len(prefix):]: z[k] for k in z.files
                        if k.startswith(prefix)})
    index = pd.read_csv(index_path) if index_path and os.path.exists(index_path) else None
    return records, index


def per_cycle_frame(record):
    """Turn a trajectory record into a tidy per-cycle DataFrame.

    Every ``m_*`` key becomes a column and the cycle index becomes the row, so
    a time series can be plotted or joined without knowing what was stored.
    """
    series = {k[2:]: np.asarray(v).ravel()
              for k, v in record.items() if k.startswith("m_")}
    if not series:
        return pd.DataFrame()
    n = max(v.size for v in series.values())
    out = {"cycle": np.arange(n)}
    for name, v in series.items():
        out[name] = v if v.size == n else np.pad(
            v.astype(float), (0, n - v.size), constant_values=np.nan)
    return pd.DataFrame(out)


def rebuild_covariances(record, taper_combine="mean"):
    """Recompute what a covariance figure shows, from one stored record.

    Returns the raw sample covariance, the taper implied by the stored radius
    field, and their product. This is the operation the panels of Figure 1
    perform, and having it here means the figure can be redrawn from the
    archive without touching the assimilation code.
    """
    from .taper import taper_matrix

    Xb = np.asarray(record["Xb"], dtype=float)
    n, N = Xb.shape
    A = (Xb - Xb.mean(axis=1, keepdims=True)) / np.sqrt(N - 1.0)
    B = A @ A.T
    r = np.asarray(record["r"], dtype=float) if "r" in record else None
    if r is None:
        return dict(B=B, C=None, BC=None)
    C = taper_matrix(n, r, combine=taper_combine)
    return dict(B=B, C=C, BC=B * C)
