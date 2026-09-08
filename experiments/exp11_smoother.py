# -*- coding: utf-8 -*-
"""
EXP-11-SMOOTHER

A criterion that works where the held-out one does not.

Everything before this experiment estimates the radius by withholding
observations. That construction has a defect which is not a matter of tuning:
the analysis at a held-out location depends only on the radius *at that
location*, so when the network is sparse the radii of unobserved points do not
enter the objective at all. Measured directly in this suite, moving the twenty
radii of unobserved points from 3 to 15 left the objective unchanged to eight
decimal places while the true error nearly doubled. Half the parameters were
invisible, which is why every spatially varying parameterization lost.

This experiment evaluates a different criterion. The state is the pair
``(k-1, k)``; the modified Cholesky factorization is estimated on the joint
vector, so its cross-block couples every component of ``k-1`` to the components
of ``k``; only the observations of ``k-1`` are assimilated; and the radius is
scored on the ``k`` block against the observations of ``k``, which that
analysis never saw. Nothing is withheld, the whole network is used, no future
data is needed, and every radius reaches the block where the score is taken.

Four things are measured.

``criterion``      Per cycle inside a running filter, does the arg-min of the
                   criterion match the arg-min of the truth, and what does the
                   difference cost? Reported against the radius the filter is
                   running at, because the failure mode of the held-out
                   criterion was precisely that its answer depended on the
                   radius that produced the ensemble.
``parameterization`` Uniform, one radius per site shared across the two times,
                   and one radius per augmented component. The middle one is
                   the interesting case: the radius describes a property of a
                   location, and that property does not change over one cycle,
                   so tying the two times halves the parameters without losing
                   anything.
``closed loop``    Letting the criterion choose, and feeding its choice into
                   the next cycle. This is where the held-out criterion
                   collapsed into a self-reinforcing fixed point, so it is the
                   test that matters.
``ablation``       The ridge scale and whether temporal predecessors are
                   allowed. The temporal rule is a design decision, not a
                   derivation, and switching it off measures what it is worth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvloc.cholesky import CholeskySpace
from cvloc.ensemble_space import Cycle
from cvloc.metaheuristics import defaults_for, get_optimizer
from cvloc.model import Lorenz96CV, observation_network
from cvloc.objective import BudgetExhausted
from cvloc.smoother import AugmentedSmoother
from common import (ExperimentContext, get_scale, latex_table, load_frozen,
                    method_allowed, parse_cli, Progress, setup_matplotlib,
                    shard_cells)

EXP_ID = "EXP-11-SMOOTHER"
DESCRIPTION = ("Augmented-state smoother on a modified Cholesky precision: a "
               "radius criterion scored on observations the analysis never "
               "assimilated.")

# How a parameter vector is spread over the 2n augmented components.
SCHEMES = {
    "uniform": lambda th, n: np.full(2 * n, float(np.atleast_1d(th)[0])),
    "per site, shared in time": lambda th, n: np.tile(np.asarray(th, float), 2),
    "per augmented component": lambda th, n: np.asarray(th, float),
}
SCHEME_K = {"uniform": 1, "per site, shared in time": 1, "per augmented component": 2}


class SmootherObjective:
    """Budget-counted wrapper around the criterion.

    The criterion is integer-valued in the radius, so candidates that floor to
    the same predecessor sets are cached and cost one evaluation.
    """

    n_evals = n_calls = n_cache_hits = 0
    stalled = False

    def __init__(self, smoother, expand, budget):
        self.sm, self.expand, self.budget = smoother, expand, budget
        self._c = {}
        self.n_evals = self.n_calls = self.n_cache_hits = 0
        self.stalled = False

    def __call__(self, theta):
        self.n_calls += 1
        if self.n_calls > self.budget * 25:
            self.stalled = True
            raise BudgetExhausted("call limit")
        key = tuple(np.floor(np.atleast_1d(theta)).astype(int))
        if key in self._c:
            self.n_cache_hits += 1
            return self._c[key]
        if self.n_evals >= self.budget:
            raise BudgetExhausted("budget")
        v = self.sm.criterion(self.expand(theta))
        self._c[key] = v
        self.n_evals += 1
        return v

    def remaining(self):
        return 0 if self.stalled else max(0, self.budget - self.n_evals)

    def best_so_far(self):
        return min(self._c.values()) if self._c else np.nan


def make_filter(scale, seed, frac, sigma=1.0):
    """A filter run, yielding the pair of ensembles at every cycle.

    The ensemble at ``k-1`` is the previous analysis propagated, not a free
    run: that is the only setting in which the criterion has to work, and the
    one an isolated pair of cycles does not reproduce.
    """
    n, N = scale.n_state, scale.ensemble_size
    rng = np.random.default_rng(seed)
    m = Lorenz96CV(n=n, F=scale.forcing, dt=scale.dt)
    T = np.array([0.0, scale.obs_freq])

    xt = m.propagate(scale.forcing * np.ones(n) + 0.01 * rng.standard_normal(n),
                     np.array([0.0, 20.0]))
    # A climatological start, with widely separated members: an ensemble whose
    # members are only lightly separated carries almost no information about
    # the relation between consecutive times, and the criterion inherits that.
    Xa = m.free_run(xt, 10.0, N)
    xt = m.propagate(xt, np.array([0.0, 5.0]))
    return m, rng, Xa, xt, T


def run_filter(scale, seed, frac, r_filter, alpha, temporal=True,
               closed_loop=False, scheme="uniform", budget=None, sigma=1.0):
    """One filter run, with the criterion either watching or steering.

    ``closed_loop=False`` runs the filter at a fixed radius and evaluates the
    criterion alongside it, which measures the criterion without letting it
    change what it measures. ``closed_loop=True`` lets the criterion choose the
    radius the next cycle runs at, which is where the held-out criterion
    collapsed.
    """
    n, N = scale.n_state, scale.ensemble_size
    m, rng, Xa, xt, T = make_filter(scale, seed, frac, sigma)
    budget = budget or scale.smoother_budget
    expand = SCHEMES[scheme]
    K = SCHEME_K[scheme] if scheme == "uniform" else (
        n if scheme == "per site, shared in time" else 2 * n)
    params = load_frozen("fpa", scale.name)

    rows = []
    Xb_prev = idx_prev = y_prev = xt_prev = None
    r_used = float(r_filter)
    err_hist = []

    for k in range(scale.smoother_cycles):
        Xb = np.stack([m.propagate(Xa[:, e], T) for e in range(N)], axis=1)
        xb = Xb.mean(axis=1)
        Xb = xb[:, None] + scale.inflation * (Xb - xb[:, None])
        xt = m.propagate(xt, T)
        idx = observation_network(n, "uniform", rng, frac)
        y = xt[idx] + sigma * rng.standard_normal(idx.size)
        ri = np.full(idx.size, 1.0 / sigma ** 2)

        record = None
        if Xb_prev is not None and k >= scale.smoother_burn:
            sm = AugmentedSmoother(Xb_prev, Xb, idx_prev, y_prev, ri_prev,
                                   idx, y, ri, alpha=alpha, r_max=scale.r_max,
                                   temporal=temporal)
            grid = np.arange(1, scale.r_max + 1, dtype=float)
            E = np.array([sm.truth_error(np.full(2 * n, r), xt_prev) for r in grid])
            J = np.array([sm.criterion(np.full(2 * n, r)) for r in grid])
            i_or, i_cv = int(E.argmin()), int(J.argmin())
            record = dict(
                cycle=k, r_truth=float(grid[i_or]), r_cv_sweep=float(grid[i_cv]),
                penalty_sweep=100.0 * (E[i_cv] / E.min() - 1.0),
                exact=int(i_or == i_cv),
                err_truth=float(E.min()), spread_pct=100.0 * (E.max() / E.min() - 1.0),
                J_spread=float(J.max() / J.min()))

            if closed_loop or scheme != "uniform":
                obj = SmootherObjective(sm, lambda t: expand(t, n), budget)
                theta, info = get_optimizer("fpa")(
                    obj, np.full(K, 1.0), np.full(K, float(scale.r_max)),
                    budget, np.random.default_rng(seed * 97 + k), **params)
                r_field = expand(theta, n)
                record.update(
                    r_search=float(np.mean(r_field)),
                    penalty_search=100.0 * (sm.truth_error(r_field, xt_prev)
                                            / E.min() - 1.0),
                    n_evals=int(obj.n_evals))
                if closed_loop:
                    r_used = float(np.mean(r_field))
            rows.append(record)

        cyc = Cycle(Xb=Xb, x_true=xt, obs_idx=idx, y=y, r_inv=ri, model=m)
        cs = CholeskySpace(cyc, ridge_alpha=0.05, r_max=scale.r_max)
        r_analysis = (np.full(n, r_used) if closed_loop
                      else np.full(n, float(r_filter)))
        _, Xa = cs.analysis_ensemble(r_analysis, rng=rng)
        err_hist.append(float(np.sqrt(np.mean((Xa.mean(axis=1) - xt) ** 2))))

        Xb_prev, idx_prev, y_prev, ri_prev, xt_prev = Xb, idx, y, ri, xt.copy()

    cut = scale.smoother_burn
    return rows, float(np.mean(err_hist[cut:])) if len(err_hist) > cut else np.nan


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(
        EXP_ID, DESCRIPTION, scale,
        extra_config=dict(alphas=list(scale.smoother_alphas),
                          r_filters=list(scale.smoother_r_filters),
                          cycles=scale.smoother_cycles,
                          schemes=list(SCHEMES)))

    ratio = 0.5 if 0.5 in scale.obs_ratios else scale.obs_ratios[-1]

    # -------- block A: criterion quality, filter at a fixed radius --------
    cells = shard_cells([(a, rf, s) for a in scale.smoother_alphas
                         for rf in scale.smoother_r_filters
                         for s in range(scale.smoother_runs)])
    rows = []
    prog = Progress(len(cells), label="runs", exp_id=EXP_ID, every=2,
                    heartbeat_s=45.0)
    for alpha, rf, s in cells:
        recs, rmse = run_filter(scale, 500 + s, ratio, rf, alpha,
                                scheme="uniform", closed_loop=False)
        for r in recs:
            rows.append(dict(exp_id=EXP_ID, block="criterion", alpha=alpha,
                             r_filter=rf, run=s, filter_rmse=rmse, **r))
        prog.step(f"alpha={alpha} filtro r={rf} corrida={s}")
    prog.done()
    dfA = pd.DataFrame(rows)
    ctx.save_table(dfA, "criterion.csv")

    # -------- block B: parameterization, and the closed loop --------
    rowsB = []
    cellsB = shard_cells([(sch, cl, s)
                          for sch in SCHEMES
                          for cl in (False, True)
                          for s in range(scale.smoother_runs)
                          if method_allowed(sch)])
    prog = Progress(len(cellsB), label="runs", exp_id=EXP_ID, every=2,
                    heartbeat_s=60.0)
    for sch, closed, s in cellsB:
        recs, rmse = run_filter(scale, 500 + s, ratio,
                                scale.smoother_r_filters[0],
                                scale.smoother_alpha, scheme=sch,
                                closed_loop=closed)
        for r in recs:
            rowsB.append(dict(exp_id=EXP_ID, block="parameterization",
                              scheme=sch, closed_loop=closed, run=s,
                              filter_rmse=rmse, **r))
        prog.step(f"{sch:<26s} lazo={'cerrado' if closed else 'abierto'} corrida={s}")
    prog.done()
    dfB = pd.DataFrame(rowsB)
    ctx.save_table(dfB, "parameterization.csv")

    # -------- ablation: temporal predecessors --------
    rowsC = []
    for temporal in (True, False):
        recs, rmse = run_filter(scale, 500, ratio, scale.smoother_r_filters[0],
                                scale.smoother_alpha, temporal=temporal,
                                scheme="uniform", closed_loop=False)
        for r in recs:
            rowsC.append(dict(exp_id=EXP_ID, block="ablation",
                              temporal=temporal, filter_rmse=rmse, **r))
    dfC = pd.DataFrame(rowsC)
    ctx.save_table(dfC, "ablation.csv")

    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(dfA) + len(dfB)))
        return

    sumA = (dfA.groupby(["alpha", "r_filter"])
            .agg(exact=("exact", "mean"),
                 penalty=("penalty_sweep", "mean"),
                 penalty_median=("penalty_sweep", "median"),
                 spread=("spread_pct", "mean"), J_spread=("J_spread", "mean"),
                 r_truth=("r_truth", "mean"), r_cv=("r_cv_sweep", "mean"),
                 n=("exact", "count")).reset_index())
    ctx.save_table(sumA, "summary.csv")
    print()
    print(sumA.to_string(index=False))

    if not dfB.empty and "penalty_search" in dfB.columns:
        sumB = (dfB.groupby(["scheme", "closed_loop"])
                .agg(penalty=("penalty_search", "mean"),
                     penalty_median=("penalty_search", "median"),
                     radius=("r_search", "mean"),
                     filter_rmse=("filter_rmse", "mean"),
                     evals=("n_evals", "mean"), n=("cycle", "count"))
                .reset_index().sort_values("penalty"))
        ctx.save_table(sumB, "summary_parameterization.csv")
        print()
        print(sumB.to_string(index=False))
    else:
        sumB = pd.DataFrame()

    if not dfC.empty:
        sumC = (dfC.groupby("temporal")
                .agg(exact=("exact", "mean"), penalty=("penalty_sweep", "mean"),
                     J_spread=("J_spread", "mean")).reset_index())
        ctx.save_table(sumC, "summary_ablation.csv")
        print()
        print(sumC.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for rf, g in dfA.groupby("r_filter"):
        axes[0].hist(g["penalty_sweep"], bins=15, alpha=0.5, label=f"filtro r={rf}")
    axes[0].set_xlabel("penalización del radio elegido (%)")
    axes[0].set_ylabel("ciclos")
    axes[0].set_title("(a) coste del criterio, por ciclo")
    axes[0].legend(fontsize=7)

    if not sumB.empty:
        order = sumB.sort_values("penalty")
        lbl = [f"{r.scheme}\n{'cerrado' if r.closed_loop else 'abierto'}"
               for r in order.itertuples()]
        axes[1].barh(range(len(order)), order["penalty"], color="tab:blue")
        axes[1].set_yticks(range(len(order)))
        axes[1].set_yticklabels(lbl, fontsize=6)
        axes[1].set_xlabel("penalización media (%)")
        axes[1].set_title("(b) parameterización y lazo cerrado")

    axes[2].scatter(dfA["r_truth"], dfA["r_cv_sweep"], s=14, alpha=0.5,
                    edgecolor="k", linewidth=0.3)
    lim = [0, scale.r_max + 1]
    axes[2].plot(lim, lim, "k:", lw=1)
    axes[2].set_xlabel("radio óptimo (verdad)")
    axes[2].set_ylabel("radio del criterio")
    axes[2].set_title("(c) acuerdo ciclo a ciclo")

    fig.suptitle(f"EXP-11: suavizado aumentado, {scale.smoother_cycles} ciclos")
    fig.tight_layout()
    ctx.save_fig(fig, "smoother.png")

    ctx.save_latex(
        latex_table(sumA, "Quality of the augmented-smoother criterion inside "
                          "a running filter.", "tab:smoother"),
        "smoother.tex")

    ctx.finish(summary=dict(
        n_cells=len(dfA) + len(dfB),
        mean_penalty=float(dfA["penalty_sweep"].mean()),
        median_penalty=float(dfA["penalty_sweep"].median()),
        exact_rate=float(dfA["exact"].mean())))


if __name__ == "__main__":
    main(parse_cli())
