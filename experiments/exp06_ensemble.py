# -*- coding: utf-8 -*-
"""
EXP-06-ENSEMBLE

Does the criterion reproduce the known dependence on ensemble size?

The optimal radius is expected to grow with N, because a larger ensemble
supports longer-range correlations before sampling noise dominates. That is not
something the criterion was designed to do, so it is an independent check
rather than a result: if the truth-based optimum grows and the criterion does
not follow, the criterion is measuring something else.

Both curves are reported over N in {10, 20, 40, 80}, together with the analysis
error that each choice produces. The criterion is expected to sit *above* the
truth-based optimum throughout, since withholding observations makes the
training network sparser than the operational one and therefore favours longer
radii; the size of that offset, and whether it is stable in N, is the useful
part of the plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvloc.cycles import select_radius, truth_optimum
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    load_frozen, parse_cli, Progress, setup_matplotlib,
                    shard_cells)

EXP_ID = "EXP-06-ENSEMBLE"
DESCRIPTION = ("Optimal radius as a function of ensemble size, truth-based "
               "and cross-validated.")


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(EXP_ID, DESCRIPTION, scale,
                            extra_config=dict(ensemble_sizes=list(scale.ensemble_sizes)))
    params = load_frozen("fpa", scale.name)

    rows = []
    cells = shard_cells([(N, ratio, ci) for N in scale.ensemble_sizes
                         for ratio in scale.obs_ratios
                         for ci in range(scale.n_cycles)])
    by_N = {}
    for N, ratio in sorted({(c[0], c[1]) for c in cells}):
        by_N[(N, ratio)] = build_spaces(scale, N=N, obs_ratio=ratio)

    prog = Progress(len(cells), label="cells", exp_id=EXP_ID, every=10)
    for N, ratio, ci in cells:
        cycles, spaces = by_N[(N, ratio)]
        sp = spaces[ci]
        to = truth_optimum(sp, scale.sweep_lo, scale.sweep_hi, scale.sweep_points)
        theta, r, obj, info = select_radius(
            sp, optimizer="fpa", param_kind="uniform", K=1,
            box_name=scale.box, folds=scale.folds, budget=scale.budget,
            seed=8000 + 101 * ci + N, opt_params=params)
        rmse_cv = sp.truth_error(r)
        rows.append(dict(
            exp_id=EXP_ID, ensemble_size=N, obs_ratio=ratio, cycle=ci,
            background_rmse=cycles[ci].background_rmse(),
            r_truth=to["r_opt"], r_cv=float(r[0]),
            log_ratio=float(np.log(r[0] / to["r_opt"])),
            rmse_truth=to["error_opt"], rmse_cv=rmse_cv,
            penalty_pct=100.0 * (rmse_cv / to["error_opt"] - 1.0),
            J=info["J"]))
        prog.step(f"N={N} p/n={ratio} cycle={ci} r_cv={r[0]:.2f}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    summary = (df.groupby(["obs_ratio", "ensemble_size"])
               .agg(r_truth=("r_truth", "mean"), r_cv=("r_cv", "mean"),
                    r_truth_sd=("r_truth", "std"), r_cv_sd=("r_cv", "std"),
                    rmse_truth=("rmse_truth", "mean"),
                    rmse_cv=("rmse_cv", "mean"),
                    penalty_pct=("penalty_pct", "mean"),
                    log_ratio=("log_ratio", "mean"),
                    background=("background_rmse", "mean"))
               .reset_index())
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    ratio0 = scale.obs_ratios[0]
    summary_all = summary
    summary = summary[summary["obs_ratio"] == ratio0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].errorbar(summary["ensemble_size"], summary["r_truth"],
                     yerr=summary["r_truth_sd"], fmt="ko-", capsize=3,
                     label="truth optimum")
    axes[0].errorbar(summary["ensemble_size"], summary["r_cv"],
                     yerr=summary["r_cv_sd"], fmt="s--", color="tab:blue",
                     capsize=3, label="CV selection")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("ensemble size $N$")
    axes[0].set_ylabel("optimal radius")
    axes[0].set_title("(a) the radius grows with $N$\nand the criterion follows it")
    axes[0].legend()

    axes[1].plot(summary["ensemble_size"], summary["rmse_truth"], "ko-",
                 label="truth optimum")
    axes[1].plot(summary["ensemble_size"], summary["rmse_cv"], "s--",
                 color="tab:blue", label="CV selection")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("ensemble size $N$")
    axes[1].set_ylabel("analysis RMSE")
    axes[1].set_title("(b) what the choice costs")
    axes[1].legend()

    fig.suptitle("EXP-06: dependence on ensemble size")
    fig.tight_layout()
    ctx.save_fig(fig, "ensemble_size.png")

    ctx.save_latex(
        latex_table(summary_all[["obs_ratio", "ensemble_size", "r_truth", "r_cv",
                             "rmse_truth", "rmse_cv", "penalty_pct"]],
                    "Optimal radius and analysis error against ensemble size.",
                    "tab:ensemble"),
        "ensemble.tex")

    ctx.finish(summary=dict(n_cells=len(df)))


if __name__ == "__main__":
    main(parse_cli())
