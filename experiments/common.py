# -*- coding: utf-8 -*-
"""
Shared infrastructure for every experiment.

Design rules that keep the suite auditable:

* Every experiment has a stable identifier (EXP-01-LANDSCAPE and so on). The
  identifier names the output directory, appears in every CSV row, and is what
  the LaTeX tables cite, so any number in the paper can be traced back to the
  run that produced it.
* Every run writes a ``manifest.json`` with the configuration, the seeds, the
  package versions and the wall time. Two runs with the same manifest must
  produce the same numbers.
* The unit of work is an assimilation **cycle** drawn from a climatological
  ensemble, not a filter run. That is the configuration of the draft, and it is
  what makes the sweeps affordable. EXP-09 is the exception and uses the full
  pyteda simulation loop.
* Three scales: ``smoke`` (a couple of minutes, validates the pipeline),
  ``quick`` (iterating on a change) and ``paper`` (what the article reports).
* No hyperparameter is tuned against the truth. EXP-00 calibrates them against
  J on held-out cycles and freezes them in ``results/frozen_params.json``;
  every other experiment reads that file when it exists.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np

warnings.filterwarnings("ignore")

import cvloc  # noqa: F401  (registers letkf-cv into pyteda)
from cvloc.cycles import CycleConfig, make_cycles
from cvloc.ensemble_space import EnsembleSpace
from cvloc.metaheuristics import defaults_for
from cvloc.progress import (Progress, announce_plan, banner, env_summary,
                            fmt_eta, log)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
FROZEN_PARAMS = os.path.join(RESULTS_ROOT, "frozen_params.json")


# ----------------------------------------------------------------------
# Sharding
# ----------------------------------------------------------------------
# The work of every experiment is a list of cells. Setting SHARD_COUNT > 1
# splits that list round-robin and each container runs only its own slice,
# writing metrics_shard<i>.csv. scripts/merge_shards.py concatenates the pieces
# and rebuilds the summaries, figures and tables.
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = max(1, int(os.environ.get("SHARD_COUNT", "1")))
METHOD_FILTER = [t.strip().lower() for t in
                 os.environ.get("METHODS", "").split(",") if t.strip()]

CYCLES_OVERRIDE = None
LIST_METHODS_ONLY = False


def method_allowed(label: str) -> bool:
    if not METHOD_FILTER:
        return True
    low = str(label).lower()
    return any(tok in low for tok in METHOD_FILTER)


def shard_suffix() -> str:
    return "" if SHARD_COUNT == 1 else f"_shard{SHARD_INDEX}"


def shard_cells(cells):
    if SHARD_COUNT == 1:
        return cells
    return [c for i, c in enumerate(cells) if i % SHARD_COUNT == SHARD_INDEX]


# ----------------------------------------------------------------------
# Scales
# ----------------------------------------------------------------------
@dataclass
class Scale:
    name: str
    # Model and cycle
    n_state: int = 40
    ensemble_size: int = 20
    forcing: float = 8.0
    dt: float = 0.01
    obs_std: float = 1.0
    spacing: float = 1.0
    spinup: float = 20.0
    n_cycles: int = 20
    seed_base: int = 42
    # Criterion and search
    folds: int = 10
    budget: int = 80
    pop_size: int = 10
    box: str = "wide"
    sweep_points: int = 60
    sweep_lo: float = 0.2
    sweep_hi: float = 20.0
    # Sweeps
    obs_ratios: tuple = (1.0, 0.5, 0.25)
    obs_stds: tuple = (0.5, 1.0, 2.0)
    ensemble_sizes: tuple = (5, 10, 20, 40)
    K_list: tuple = (1, 2, 4, 8, 20, 40)
    budgets: tuple = (20, 40, 80, 160, 320)
    pop_sizes: tuple = (10, 20, 50, 100)
    gammas: tuple = (0.01, 0.05, 0.1, 0.5, 1.0)
    boxes: tuple = ("wide", "informed", "tight")
    fold_counts: tuple = (2, 5, 10, 20)
    n_repeats: int = 3          # independent search seeds per cell
    n_tuning_cycles: int = 8    # held-out cycles for EXP-00
    budget_scale_cap: float = 2.0   # ceiling on the K-scaled budget
    # Snapshot persistence. Fractions of a multi-cycle run at which the
    # forecast and analysis ensembles are stored, so a covariance panel can be
    # redrawn without rerunning the assimilation. None disables it.
    store_states_at: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Modified Cholesky
    r_max: int = 12
    ridge_alpha: float = 0.01
    # Multi-cycle (EXP-09), now the headline regime
    n_scenarios: int = 4
    schedules: tuple = ("fixed", "random")
    smoothings: tuple = ("none", "window", "ewma", "freeze")
    obs_freq: float = 0.1
    # 300 cycles. The leading Lyapunov exponent of Lorenz 96 at F = 8 is near
    # 1.68 per time unit, so the e-folding time is about 0.6 and, at an
    # observation frequency of 0.1, consecutive cycles are correlated over
    # roughly six steps. With 30% burn-in, 300 cycles leave about 35 effective
    # samples per scenario, which is what it takes to resolve differences of a
    # few percent between methods. The 60 cycles used previously left seven.
    end_time: float = 30.0
    inflation: float = 1.04
    burn_in_frac: float = 0.30
    pert_xb: float = 0.5
    pert_ensemble: float = 0.05
    spinup_xb: float = 5.0
    spinup_ensemble: float = 5.0

    @property
    def cycle_seeds(self):
        return [self.seed_base + 13 * i for i in range(self.n_cycles)]

    @property
    def tuning_seeds(self):
        # Disjoint from cycle_seeds by construction: EXP-00 must not calibrate
        # on the cycles the other experiments report.
        return [900_000 + 13 * i for i in range(self.n_tuning_cycles)]


SCALES = {
    "smoke": Scale(name="smoke", n_cycles=3, budget=30, sweep_points=20,
                   obs_ratios=(1.0, 0.5), obs_stds=(0.5, 1.0),
                   ensemble_sizes=(10, 20), K_list=(1, 4, 40),
                   budgets=(15, 30), pop_sizes=(10, 20),
                   gammas=(0.01, 0.1, 1.0), boxes=("wide", "informed"),
                   fold_counts=(2, 10), n_repeats=2, n_tuning_cycles=2,
                   n_scenarios=2, smoothings=("none", "freeze"),
                   end_time=2.0),
    "quick": Scale(name="quick", n_cycles=8, budget=60, sweep_points=40,
                   obs_ratios=(1.0, 0.5, 0.25), obs_stds=(0.5, 1.0, 2.0),
                   ensemble_sizes=(5, 10, 20, 40), K_list=(1, 2, 4, 8, 40),
                   budgets=(20, 60, 160), pop_sizes=(10, 20, 50),
                   n_repeats=2, n_tuning_cycles=4, n_scenarios=3,
                   end_time=10.0),
    "paper": Scale(name="paper"),
}


def get_scale(name=None) -> Scale:
    name = name or os.environ.get("SCALE", "smoke")
    if name not in SCALES:
        raise ValueError(f"unknown scale '{name}'. Available: {sorted(SCALES)}")
    scale = SCALES[name]
    if CYCLES_OVERRIDE:
        scale = replace(scale, n_cycles=CYCLES_OVERRIDE)
    return scale


def parse_cli(argv=None):
    """Parse the command line of an experiment script.

    Everything can also be set through environment variables, which is what the
    Docker images use; an explicit flag always wins over the environment.
    """
    import argparse

    global SHARD_INDEX, SHARD_COUNT, METHOD_FILTER, CYCLES_OVERRIDE
    global LIST_METHODS_ONLY

    ap = argparse.ArgumentParser(
        description="Run one experiment of the CV-localization suite.")
    ap.add_argument("scale", nargs="?", default=None,
                    help="smoke, quick or paper (default: $SCALE, then smoke)")
    ap.add_argument("--methods", "-m", default=None,
                    help="comma-separated substrings; only matching methods run")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--cycles", type=int, default=None,
                    help="override the number of independent cycles")
    ap.add_argument("--list-methods", action="store_true")
    args = ap.parse_args(argv)

    if args.methods is not None:
        METHOD_FILTER = [t.strip().lower()
                         for t in args.methods.split(",") if t.strip()]
    if args.shard is not None:
        SHARD_INDEX = int(args.shard)
    if args.shards is not None:
        SHARD_COUNT = max(1, int(args.shards))
    CYCLES_OVERRIDE = args.cycles
    LIST_METHODS_ONLY = bool(args.list_methods)
    return args.scale


def maybe_list_methods(methods, exp_id) -> bool:
    if not LIST_METHODS_ONLY:
        return False
    print(f"\n{exp_id} would run {len(methods)} methods:")
    for label in methods:
        mark = "  " if method_allowed(label) else "  (filtered out) "
        print(f"{mark}{label}")
    print()
    return True


# ----------------------------------------------------------------------
# Cycles
# ----------------------------------------------------------------------
def network_for(obs_ratio):
    """Map an observation ratio to a network kind and density."""
    if obs_ratio >= 1.0:
        return "full", 1.0
    return "uniform", float(obs_ratio)


def cycle_config(scale: Scale, N=None, forcing="uniform", network="full",
                 obs_ratio=None, **over) -> CycleConfig:
    if obs_ratio is not None:
        network, dense = network_for(obs_ratio)
        over.setdefault("dense_frac", dense)
    cfg = CycleConfig(
        n=scale.n_state,
        N=N or scale.ensemble_size,
        F=scale.forcing,
        dt=scale.dt,
        forcing=forcing,
        network=network,
        obs_std=scale.obs_std,
        spinup=scale.spinup,
        spacing=scale.spacing,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def build_spaces(scale: Scale, N=None, forcing="uniform", network="full",
                 obs_ratio=None, seeds=None, verbose=True, **over):
    """Draw independent cycles and precompute their ensemble-space quantities.

    Returns ``(cycles, spaces)``. Each cycle has its own free run, so the
    cycle-to-cycle spread reported in the figures is a genuine spread over the
    attractor and not a spread over observation noise alone.
    """
    cfg = cycle_config(scale, N=N, forcing=forcing, network=network,
                       obs_ratio=obs_ratio, **over)
    seeds = list(seeds) if seeds is not None else scale.cycle_seeds
    cycles = [__import__("cvloc").make_cycle(cfg, s) for s in seeds]
    spaces = [EnsembleSpace(c) for c in cycles]
    if verbose:
        c0 = cycles[0]
        log(f"built {len(cycles)} cycles: N={c0.N}  p={c0.p}/{c0.n}  "
            f"forcing={forcing}  network={cfg.network}  "
            f"background RMSE={np.mean([c.background_rmse() for c in cycles]):.2f}")
    return cycles, spaces


# ----------------------------------------------------------------------
# Frozen hyperparameters
# ----------------------------------------------------------------------
def load_frozen(optimizer: str, scale_name: str = None) -> dict:
    """Hyperparameters for an optimizer: calibrated if available, else default.

    EXP-00 writes ``results/frozen_params.json``. Reading it here rather than
    hard-coding the numbers means a re-calibration propagates to every
    experiment without touching their code, and that the paper can state that
    no constant was chosen by looking at the truth.
    """
    params = defaults_for(optimizer)
    if os.path.exists(FROZEN_PARAMS):
        try:
            with open(FROZEN_PARAMS) as fh:
                frozen = json.load(fh)
            block = frozen.get(scale_name, frozen.get("params", frozen))
            if optimizer in block:
                params.update(block[optimizer])
                log(f"using calibrated parameters for {optimizer}: "
                    f"{block[optimizer]}")
        except Exception as exc:
            log(f"could not read {FROZEN_PARAMS}: {exc}", level="WARN")
    return params


def save_frozen(mapping: dict, scale_name: str):
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    existing = {}
    if os.path.exists(FROZEN_PARAMS):
        try:
            with open(FROZEN_PARAMS) as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}
    existing[scale_name] = mapping
    existing["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(FROZEN_PARAMS, "w") as fh:
        json.dump(existing, fh, indent=2, default=str)
    log(f"wrote {FROZEN_PARAMS}")


# ----------------------------------------------------------------------
# Output management
# ----------------------------------------------------------------------
class ExperimentContext:
    """Creates the output directory, writes the manifest, saves the tables."""

    def __init__(self, exp_id, description, scale: Scale, extra_config=None):
        self.exp_id = exp_id
        self.description = description
        self.scale = scale
        self.extra_config = dict(extra_config or {})
        self.dir = os.path.join(RESULTS_ROOT, f"{exp_id}_{scale.name}")
        self.fig_dir = os.path.join(self.dir, "figures")
        self.tab_dir = os.path.join(self.dir, "tables")
        for d in (self.dir, self.fig_dir, self.tab_dir):
            os.makedirs(d, exist_ok=True)
        self.t0 = time.time()
        banner(f"{exp_id}   [scale={scale.name}]",
               [description, f"output: {self.dir}"])
        env_summary()

    def path(self, name):
        return os.path.join(self.dir, name)

    @property
    def is_shard(self) -> bool:
        return SHARD_COUNT > 1

    def save_table(self, df, name):
        if self.is_shard:
            stem, ext = os.path.splitext(name)
            name = f"{stem}{shard_suffix()}{ext}"
        p = os.path.join(self.dir, name)
        df.to_csv(p, index=False)
        log(f"wrote {name}  ({len(df)} rows)", self.exp_id)
        return p

    def save_latex(self, text, name):
        if self.is_shard:
            return None
        p = os.path.join(self.tab_dir, name)
        with open(p, "w") as fh:
            fh.write(text)
        log(f"wrote tables/{name}", self.exp_id)
        return p

    def save_snapshots(self, writer, name="snapshots"):
        """Write the ensemble archive plus its index.

        Shards write their own archive; merge_shards leaves them side by side
        rather than concatenating, since an npz of a few hundred megabytes is
        better read a piece at a time.
        """
        if writer is None or len(writer) == 0:
            return None
        stem = f"{name}{shard_suffix()}"
        path = writer.write(self.path(f"{stem}.npz"),
                            self.path(f"{stem}_index.csv"))
        if path:
            size_mb = os.path.getsize(path) / 1e6
            log(f"wrote {stem}.npz  ({len(writer)} records, {size_mb:.1f} MB)",
                self.exp_id)
        return path

    def save_fig(self, fig, name, dpi=130):
        if self.is_shard:
            log(f"skipping figure {name}, a shard has partial data only",
                self.exp_id)
            return None
        p = os.path.join(self.fig_dir, name)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        log(f"wrote figures/{name}", self.exp_id)
        return p

    def finish(self, summary=None):
        import pyteda

        manifest = dict(
            exp_id=self.exp_id,
            description=self.description,
            model="Lorenz96",
            scale=asdict(self.scale),
            extra_config=self.extra_config,
            shard=dict(index=SHARD_INDEX, count=SHARD_COUNT,
                       method_filter=METHOD_FILTER),
            elapsed_s=round(time.time() - self.t0, 2),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            versions=dict(
                cvloc=cvloc.__version__,
                pyteda=getattr(pyteda, "__version__", "unknown"),
                numpy=np.__version__,
                python=sys.version.split()[0],
                platform=platform.platform(),
            ),
            env=dict(
                PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED"),
                OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
            ),
        )
        if summary is not None:
            manifest["summary"] = summary
        p = self.path(f"manifest{shard_suffix()}.json")
        with open(p, "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        log("wrote manifest.json", self.exp_id)
        log(f"DONE in {fmt_eta(manifest['elapsed_s'])}", self.exp_id)


def bail_if_empty(rows, ctx) -> bool:
    """True when the selection left no work, so the caller should return."""
    if len(rows):
        return False
    log("no cells matched the current selection "
        f"(methods={METHOD_FILTER or 'all'}, "
        f"shard {SHARD_INDEX + 1}/{SHARD_COUNT}); nothing to do",
        ctx.exp_id, level="WARN")
    ctx.finish(summary=dict(skipped=True, method_filter=METHOD_FILTER,
                            shard=[SHARD_INDEX, SHARD_COUNT]))
    return True


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def method_seed(exp_id, label, cycle_id, repeat=0, base=1000):
    """Deterministic seed from (experiment, method, cycle, repeat)."""
    h = abs(hash((exp_id, str(label), int(cycle_id), int(repeat)))) % 10_000_019
    return base + h


def aggregate(df, by="method", value="rmse"):
    g = df.groupby(by)[value]
    out = g.agg(["mean", "std", "min", "max", "count"]).reset_index()
    return out.sort_values("mean")


def fmt_mean_std(mean, std):
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.4f} ({std:.4f})" if np.isfinite(std) else f"{mean:.4f}"


def latex_table(df, caption, label, float_fmt="%.4f") -> str:
    body = df.to_latex(index=False, escape=True, float_format=float_fmt)
    return (f"% generated by the CV-localization suite\n"
            f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n{body}\\end{{table}}\n")


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
    })
    return plt
