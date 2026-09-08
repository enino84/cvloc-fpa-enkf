# -*- coding: utf-8 -*-
"""
EXP-10-CHOLESKY

The radius as a predecessor set, and the comparison that finally discriminates.

In the tapered formulation the radius enters through a smooth kernel and the
objective inherits that smoothness. On one free radius the landscape is an
interval with a single basin, and eighty evaluations find its minimum whatever
the search. Reporting a tie there is honest but it is not evidence for
anything, and a paper co-authored with the author of the algorithm cannot rest
on it.

Here the radius determines which components enter a regression rather than the
width of a taper,

    x_[i] regressed on { j < i : d(i,j) <= r_i },  B^{-1} = L' D^{-1} L

so the estimator changes only when a radius crosses an integer. The landscape
is piecewise constant with genuine plateaus rather than a noise floor, it is
not unimodal, and a search that merely walks downhill has something to fail at.

Three things are done.

* The **exhaustive sweep** over integer radii, which for one free radius is
  affordable and gives the exact minimizer. Every search is then reported as a
  gap to a known optimum rather than as a comparison against the other
  searches, which is a much stronger form of the claim.
* The **head-to-head** at equal budget, on one free radius and on several.
* A **structural diagnostic** of the landscape itself: the number of local
  minima along the integer sweep and the width of the plateaus, which is what
  makes this problem different and should be stated rather than asserted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvloc.cholesky import CholeskySpace
from cvloc.metaheuristics import CONTROLS, LABELS, METAHEURISTICS, get_optimizer
from cvloc.objective import CVObjective
from cvloc.parameterization import make as make_param
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    load_frozen, maybe_list_methods, method_allowed,
                    parse_cli, Progress, setup_matplotlib, shard_cells)

EXP_ID = "EXP-10-CHOLESKY"
DESCRIPTION = ("Modified Cholesky parameterization: the radius as a "
               "predecessor set, on a discrete and multimodal landscape.")


def _boxplot(ax, data, labels, **kw):
    """Horizontal boxplot that works across matplotlib versions.

    ``labels`` was renamed to ``tick_labels`` in matplotlib 3.9 and removed
    outright afterwards, and ``vert`` is on the same path. Passing the old
    names raises a TypeError on a current install, which is how this surfaced:
    the figure crashed at the end of a run whose data had already been saved.
    """
    try:
        return ax.boxplot(data, tick_labels=labels, orientation="horizontal",
                          **kw)
    except TypeError:
        pass
    try:
        return ax.boxplot(data, tick_labels=labels, vert=False, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, vert=False, **kw)



def integer_sweep(chol_space, r_max, folds, fold_seed):
    """Exhaustive sweep over the integer radii, with truth and CV curves."""
    param = make_param("uniform", chol_space.n)
    obj = CVObjective(chol_space, param=param, folds=folds,
                      rng=np.random.default_rng(fold_seed), budget=None,
                      quantize=None)
    radii = np.arange(1, r_max + 1, dtype=float)
    cost = np.array([obj.raw(np.array([r])) for r in radii])
    err = np.array([chol_space.truth_error(np.full(chol_space.n, r))
                    for r in radii])
    return radii, cost, err


def count_local_minima(y):
    """Interior strict local minima of a sequence."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return 0
    return int(np.sum((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])))


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(EXP_ID, DESCRIPTION, scale,
                            extra_config=dict(r_max=scale.r_max,
                                              ridge_alpha=scale.ridge_alpha))

    methods = METAHEURISTICS + ["random"]
    if maybe_list_methods({LABELS[m]: m for m in methods}, EXP_ID):
        return

    # The observation error sets the balance between what the background and
    # the observations contribute, and therefore how far the analysis needs to
    # reach. It is the axis a referee will ask about and it is nearly free
    # here, so the sweep runs on the exhaustive part.
    chol_by_std = {}
    for so in scale.obs_stds:
        cyc, _ = build_spaces(scale, obs_std=so, verbose=(so == scale.obs_stds[0]))
        chol_by_std[so] = [CholeskySpace(c, ridge_alpha=scale.ridge_alpha,
                                         r_max=scale.r_max) for c in cyc]
    std0 = 1.0 if 1.0 in scale.obs_stds else scale.obs_stds[0]
    chol = chol_by_std[std0]

    # ---------------- the landscape itself ----------------
    sweep_rows, curves = [], []
    for so in scale.obs_stds:
        for ci, cs in enumerate(chol_by_std[so]):
            radii, cost, err = integer_sweep(cs, scale.r_max, scale.folds,
                                             20260828)
            if so == std0:
                curves.append((cost, err))
            i_cv, i_or = int(cost.argmin()), int(err.argmin())
            sweep_rows.append(dict(
                exp_id=EXP_ID, obs_std=so, cycle=ci, r_cv=float(radii[i_cv]),
                r_truth=float(radii[i_or]), cost_opt=float(cost[i_cv]),
                err_cv=float(err[i_cv]), err_truth=float(err[i_or]),
                penalty_pct=100.0 * (err[i_cv] / err[i_or] - 1.0),
                local_minima_cv=count_local_minima(cost),
                local_minima_err=count_local_minima(err),
                spread_pct=100.0 * (err.max() / err.min() - 1.0)))
    sweep = pd.DataFrame(sweep_rows)
    ctx.save_table(sweep, "sweep.csv")
    by_std = (sweep.groupby("obs_std")
              .agg(r_cv=("r_cv", "mean"), r_truth=("r_truth", "mean"),
                   penalty=("penalty_pct", "mean"),
                   local_minima=("local_minima_cv", "mean"),
                   spread=("spread_pct", "mean")).reset_index())
    ctx.save_table(by_std, "by_obs_std.csv")
    print()
    print(by_std.to_string(index=False))

    # The exact minimum per cycle, used as the reference for the gap.
    exact = {r["cycle"]: r["cost_opt"] for r in sweep_rows if r["obs_std"] == std0}

    # ---------------- head to head ----------------
    K_multi = max(2, min(4, max(scale.K_list)))
    blocks = [("uniform", 1), ("blocks", K_multi)]
    cells = shard_cells([(opt, kind, K, ci, rep)
                         for opt in methods
                         for kind, K in blocks
                         for ci in range(len(chol))
                         for rep in range(scale.n_repeats)
                         if method_allowed(LABELS[opt])])

    rows = []
    prog = Progress(len(cells), label="cells", exp_id=EXP_ID, every=20)
    for opt, kind, K, ci, rep in cells:
        cs = chol[ci]
        param = make_param(kind, cs.n, K=K)
        # The radius is an integer count of grid points here, so the box is in
        # integers and the cache key is the rounded vector: candidates that
        # round to the same predecessor sets cost one evaluation, not several.
        lo = np.full(K, 1.0)
        hi = np.full(K, float(scale.r_max))
        obj = CVObjective(cs, param=param, folds=scale.folds,
                          rng=np.random.default_rng(20260828),
                          budget=scale.budget, quantize=None)
        params = load_frozen(opt, scale.name)
        theta, info = get_optimizer(opt)(
            obj, lo, hi, scale.budget,
            np.random.default_rng(13000 + 31 * rep + 101 * ci), **params)
        r = param.expand(theta)
        rows.append(dict(
            exp_id=EXP_ID, optimizer=opt, method=LABELS[opt],
            is_control=opt in CONTROLS, param=kind, K=K, cycle=ci, repeat=rep,
            J=info["J"], rmse=cs.truth_error(r),
            radius_mean=float(np.mean(np.floor(r))),
            n_evals=info["n_evals"], n_cache_hits=info["n_cache_hits"],
            J_exact=exact[ci] if K == 1 else np.nan))
        prog.step(f"{LABELS[opt]:<22s} K={K} cycle={ci} J={info['J']:.4f}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    # For K = 1 the exhaustive minimum is known, so the gap is to the truth of
    # the optimization problem, not to the best competitor.
    df["J_ref"] = df["J_exact"]
    fallback = df.groupby(["K", "cycle"])["J"].transform("min")
    df["J_ref"] = df["J_ref"].fillna(fallback)
    df["J_gap_pct"] = 100.0 * (df["J"] / df["J_ref"] - 1.0)
    df["solved_exactly"] = (df["J_gap_pct"] <= 1e-9) & df["J_exact"].notna()

    summary = (df.groupby(["K", "method", "is_control"])
               .agg(J=("J", "mean"), J_gap_pct=("J_gap_pct", "mean"),
                    J_gap_sd=("J_gap_pct", "std"),
                    solved=("solved_exactly", "mean"),
                    rmse=("rmse", "mean"), radius=("radius_mean", "mean"),
                    evals=("n_evals", "mean"),
                    cache_hits=("n_cache_hits", "mean"), n=("J", "count"))
               .reset_index().sort_values(["K", "J_gap_pct"]))
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"  landscape: {sweep['local_minima_cv'].mean():.1f} interior local "
          f"minima of J on average over {scale.r_max} integer radii")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    C = np.array([c for c, _ in curves])
    E = np.array([e for _, e in curves])
    radii = np.arange(1, scale.r_max + 1)
    axes[0].plot(radii, E.mean(0), "ko-", ms=3, label="RMSE (truth)")
    axes[0].set_xlabel("predecessor radius $r$")
    axes[0].set_ylabel("analysis RMSE")
    ax2 = axes[0].twinx()
    ax2.plot(radii, C.mean(0), "s--", color="tab:blue", ms=3)
    ax2.set_ylabel("CV cost", color="tab:blue")
    ax2.grid(False)
    axes[0].set_title("(a) the discrete landscape\nis not unimodal")
    axes[0].legend(fontsize=7)

    for K, ax in zip([b[1] for b in blocks], axes[1:]):
        sub = df[df["K"] == K]
        order = (sub.groupby("method")["J_gap_pct"].mean()
                 .sort_values().index.tolist())
        data = [sub[sub["method"] == m]["J_gap_pct"].values for m in order]
        bp = _boxplot(ax, data, order, patch_artist=True, widths=0.6)
        for patch, m in zip(bp["boxes"], order):
            ctrl = bool(sub[sub["method"] == m]["is_control"].iloc[0])
            patch.set_facecolor("lightgrey" if ctrl else "tab:green")
            patch.set_alpha(0.75)
        ref = "exact optimum" if K == 1 else "best found"
        ax.set_xlabel(f"gap to the {ref} (%)")
        ax.set_title(f"K = {K} free radii")

    fig.suptitle(f"EXP-10: modified Cholesky, budget {scale.budget}")
    fig.tight_layout()
    ctx.save_fig(fig, "cholesky.png")

    ctx.save_latex(
        latex_table(summary.drop(columns=["is_control", "J_gap_sd"]),
                    "Searches on the modified Cholesky parameterization, where "
                    "the radius sets the predecessor set of a regression. For "
                    "$K=1$ the exhaustive optimum is known and the gap is "
                    "absolute.", "tab:cholesky"),
        "cholesky.tex")

    ctx.finish(summary=dict(
        n_cells=len(df),
        mean_local_minima=float(sweep[sweep["obs_std"] == std0]["local_minima_cv"].mean()),
        mean_penalty_pct=float(sweep["penalty_pct"].mean())))


if __name__ == "__main__":
    main(parse_cli())
