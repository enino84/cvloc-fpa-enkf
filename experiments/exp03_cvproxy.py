# -*- coding: utf-8 -*-
"""
EXP-03-CVPROXY

Is the cross-validated criterion a good stand-in for the truth?

The whole method rests on selecting the radius by minimizing J, which never
sees the true state. This is the experiment most likely to falsify it, so it is
worth reading before anything else.

Three things are measured.

* The **penalty**: the error actually incurred by the radius the criterion
  selects, relative to the truth-based optimum of the *same* cycle. This is the
  quantity that matters operationally, and its distribution is reported rather
  than agreement between arg-minima, because the error curve is flat towards
  short radii and the truth-based arg-min is therefore weakly identified inside
  that flat region while the error difference across it is negligible.

* The **bias**: whether the criterion systematically selects a longer radius
  than the oracle. It should. Withholding a fold makes the training network
  sparser than the operational one, so the analysis it scores wants to reach
  further than the analysis that will actually be run. If the bias is there and
  is stable, it can be corrected; if it is there and is erratic, it cannot.

* The **sensitivity to the number of folds**. More folds means less is withheld
  per fold and the mismatch above shrinks, but also that each score is noisier.
  The sweep says where that trade-off sits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from cvloc.cycles import (cv_sweep, select_radius, select_radius_bagged,
                          truth_optimum)
from common import load_frozen
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    parse_cli, Progress, setup_matplotlib, shard_cells)

EXP_ID = "EXP-03-CVPROXY"
DESCRIPTION = ("Agreement between the cross-validated criterion and the "
               "truth-based error: observation density, fold count, and the "
               "averaged against the bagged estimator.")


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(EXP_ID, DESCRIPTION, scale,
                            extra_config=dict(fold_counts=list(scale.fold_counts)))

    built = {ratio: build_spaces(scale, obs_ratio=ratio)
             for ratio in scale.obs_ratios}
    cells = shard_cells([(m, ratio, ci) for m in scale.fold_counts
                         for ratio in scale.obs_ratios
                         for ci in range(scale.n_cycles)])

    rows = []
    prog = Progress(len(cells), label="cells", exp_id=EXP_ID, every=10)
    for m_folds, ratio, ci in cells:
        cycles, spaces = built[ratio]
        sp = spaces[ci]
        to = truth_optimum(sp, scale.sweep_lo, scale.sweep_hi, scale.sweep_points)
        cs = cv_sweep(sp, folds=m_folds, lo=scale.sweep_lo, hi=scale.sweep_hi,
                      n_points=scale.sweep_points)

        err, cost, grid = to["error"], cs["cost"], to["grid"]
        i_cv = int(cost.argmin())
        i_or = int(err.argmin())
        err_cv = float(err[i_cv])

        # Rank of the CV choice in the truth ordering: 1 means the criterion
        # picked the best available radius.
        rank = int(np.argsort(np.argsort(err))[i_cv]) + 1

        rows.append(dict(
            exp_id=EXP_ID, folds=m_folds, obs_ratio=ratio, p_obs=int(sp.p),
            cycle=ci,
            r_cv=float(grid[i_cv]), r_truth=float(grid[i_or]),
            log_ratio=float(np.log(grid[i_cv] / grid[i_or])),
            err_cv=err_cv, err_truth=float(err[i_or]),
            err_worst=float(err.max()),
            penalty_pct=100.0 * (err_cv / err[i_or] - 1.0),
            rank_of_cv=rank, n_candidates=grid.size,
            same_argmin=int(i_cv == i_or),
            spearman=float(spearmanr(cost, err).statistic),
            kendall=float(kendalltau(cost, err).statistic),
        ))
        prog.step(f"folds={m_folds} p/n={ratio} cycle={ci} "
                  f"penalty={rows[-1]['penalty_pct']:.1f}%")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")

    # ------------------------------------------------------------------
    # Averaged against bagged estimator, at the main fold count.
    #
    # The criterion as written in the draft averages J over the folds and
    # minimizes the average. The alternative redraws the split B times,
    # minimizes each realization, and takes the mode of the arg-minima. On a
    # flat landscape a mode only needs each vote in the right bin, not the
    # exact minimum, so it should degrade more gracefully; whether it does
    # here, and at what cost in evaluations, is the question.
    params = load_frozen("fpa", scale.name)
    est_rows = []
    est_cells = shard_cells([(ratio, ci) for ratio in scale.obs_ratios
                             for ci in range(scale.n_cycles)])
    prog = Progress(len(est_cells) * 3, label="estimators", exp_id=EXP_ID,
                    every=10)
    for ratio, ci in est_cells:
        cycles, spaces = built[ratio]
        sp = spaces[ci]
        to = truth_optimum(sp, scale.sweep_lo, scale.sweep_hi, scale.sweep_points)
        B = max(3, scale.folds + 5)

        for name, fn in (("averaged", "mean"), ("bagged, cold", "cold"),
                         ("bagged, warm", "warm")):
            if fn == "mean":
                _, r, obj, info = select_radius(
                    sp, optimizer="fpa", box_name=scale.box, folds=scale.folds,
                    budget=scale.budget, seed=400 + ci, opt_params=params)
                evals, iqr = info["n_evals"], float("nan")
            else:
                _, r, votes, info = select_radius_bagged(
                    sp, optimizer="fpa", box_name=scale.box,
                    folds=scale.folds, budget=(scale.budget if fn == "cold"
                                               else max(10, scale.budget // 3)),
                    n_resamples=B, warm=(fn == "warm"), seed=400 + ci,
                    opt_params=params)
                evals, iqr = info["n_evals"], info["vote_iqr"]
            rmse = sp.truth_error(r)
            est_rows.append(dict(
                exp_id=EXP_ID, estimator=name, obs_ratio=ratio, cycle=ci,
                radius=float(np.mean(r)), r_truth=to["r_opt"], rmse=rmse,
                penalty_pct=100.0 * (rmse / to["error_opt"] - 1.0),
                n_evals=evals, vote_iqr=iqr, n_resamples=B))
            prog.step(f"{name:<14s} p/n={ratio} cycle={ci}")
    prog.done()
    est = pd.DataFrame(est_rows)
    ctx.save_table(est, "estimators.csv")
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    by_ratio = (df[df["folds"] == scale.folds].groupby("obs_ratio")
                .agg(p_obs=("p_obs", "mean"),
                     penalty_mean=("penalty_pct", "mean"),
                     penalty_median=("penalty_pct", "median"),
                     r_cv=("r_cv", "mean"), r_truth=("r_truth", "mean"),
                     spearman=("spearman", "mean")).reset_index())
    ctx.save_table(by_ratio, "by_ratio.csv")
    print()
    print(by_ratio.to_string(index=False))

    if not est.empty:
        by_est = (est.groupby(["obs_ratio", "estimator"])
                  .agg(penalty_mean=("penalty_pct", "mean"),
                       penalty_median=("penalty_pct", "median"),
                       radius=("radius", "mean"), evals=("n_evals", "mean"),
                       iqr=("vote_iqr", "mean")).reset_index())
        ctx.save_table(by_est, "by_estimator.csv")
        print()
        print(by_est.to_string(index=False))

    by_folds = (df.groupby("folds")
                .agg(penalty_mean=("penalty_pct", "mean"),
                     penalty_median=("penalty_pct", "median"),
                     penalty_p90=("penalty_pct", lambda s: float(np.percentile(s, 90))),
                     agreement=("same_argmin", "mean"),
                     spearman=("spearman", "mean"),
                     log_ratio=("log_ratio", "mean"),
                     r_cv=("r_cv", "mean"), r_truth=("r_truth", "mean"))
                .reset_index())
    ctx.save_table(by_folds, "summary.csv")
    print()
    print(by_folds.to_string(index=False))

    main_folds = df[df["folds"] == scale.folds]
    if main_folds.empty:
        main_folds = df[df["folds"] == df["folds"].max()]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].hist(main_folds["penalty_pct"], bins=12, color="tab:red",
                 edgecolor="k", alpha=0.85)
    axes[0].axvline(main_folds["penalty_pct"].mean(), color="k", ls="--",
                    label=f"mean {main_folds['penalty_pct'].mean():.1f}%")
    axes[0].set_xlabel("RMSE penalty of the selected radius (%)")
    axes[0].set_ylabel("number of cycles")
    axes[0].set_title("(a) cost of using the CV radius\ninstead of the oracle")
    axes[0].legend()

    axes[1].scatter(main_folds["r_truth"], main_folds["r_cv"], s=26,
                    edgecolor="k", color="tab:blue")
    lim = [min(main_folds["r_truth"].min(), main_folds["r_cv"].min()) * 0.8,
           max(main_folds["r_truth"].max(), main_folds["r_cv"].max()) * 1.2]
    axes[1].plot(lim, lim, "k:", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("truth-based radius")
    axes[1].set_ylabel("CV-selected radius")
    axes[1].set_title("(b) the criterion favours longer radii")

    axes[2].errorbar(by_folds["folds"], by_folds["penalty_mean"],
                     yerr=None, fmt="o-", color="tab:green",
                     label="mean penalty")
    axes[2].plot(by_folds["folds"], by_folds["penalty_median"], "s--",
                 color="tab:olive", label="median penalty")
    axes[2].set_xlabel("number of folds $m$")
    axes[2].set_ylabel("RMSE penalty (%)")
    axes[2].set_title("(c) sensitivity to the fold count")
    axes[2].legend()

    fig.suptitle("EXP-03: the criterion against the truth it is not allowed to see")
    fig.tight_layout()
    ctx.save_fig(fig, "cv_proxy.png")

    ctx.save_latex(
        latex_table(by_folds, "Quality of the cross-validated criterion as a "
                              "proxy for the truth-based optimum.",
                    "tab:cvproxy"),
        "cvproxy.tex")

    ctx.finish(summary=dict(
        n_cells=len(df),
        mean_penalty_pct=float(main_folds["penalty_pct"].mean()),
        median_penalty_pct=float(main_folds["penalty_pct"].median()),
        mean_spearman=float(main_folds["spearman"].mean())))


if __name__ == "__main__":
    main(parse_cli())
