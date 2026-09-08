# -*- coding: utf-8 -*-
"""
EXP-04-META

How do the searches compare at equal budget, on the tapered parameterization?

Every optimizer minimizes the same J, over the same frozen folds, from the same
ensemble, with the same number of unique evaluations. Random sampling, a dense
grid sweep and Nelder-Mead are included not as competitors but as the
calibration of the comparison: a metaheuristic that does not beat random
sampling at the same budget is not doing anything, and on a smooth
one-dimensional box a grid is often the honest answer.

A word on what to expect, because it is better stated than discovered by a
referee. With one free radius the objective is a smooth function on an interval
and eighty evaluations are plenty; the searches should tie, and reporting that
tie is the correct outcome. The comparison becomes informative in two places
only: the multi-radius case, run here as a second block, and the discrete
landscape of EXP-10. Presenting a tie on the easy problem as evidence for one
algorithm would not survive review, and this experiment is arranged so that it
cannot be read that way.

The gap to the best value any method found on the cycle is reported alongside
the raw attained cost, since on a flat landscape absolute differences in J are
uninformative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvloc.cycles import select_radius, truth_optimum
from cvloc.metaheuristics import CONTROLS, LABELS, METAHEURISTICS
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    load_frozen, maybe_list_methods, method_allowed,
                    parse_cli, Progress, setup_matplotlib, shard_cells)

EXP_ID = "EXP-04-META"
DESCRIPTION = ("Head-to-head between searches at equal budget on the tapered "
               "parameterization, with random, grid and local-search controls.")


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(EXP_ID, DESCRIPTION, scale,
                            extra_config=dict(budget=scale.budget))

    methods = METAHEURISTICS + CONTROLS
    if maybe_list_methods({LABELS[m]: m for m in methods}, EXP_ID):
        return

    cycles, spaces = build_spaces(scale)
    K_multi = max(K for K in scale.K_list if K <= 8)
    blocks = [("uniform", 1), ("blocks", K_multi)]

    cells = shard_cells([(opt, kind, K, ci, rep)
                         for opt in methods
                         for kind, K in blocks
                         for ci in range(len(spaces))
                         for rep in range(scale.n_repeats)
                         if method_allowed(LABELS[opt])])

    # The truth-based optimum per cycle, computed once, as a reference column.
    oracle = {ci: truth_optimum(spaces[ci], scale.sweep_lo, scale.sweep_hi,
                                scale.sweep_points)
              for ci in sorted({c[3] for c in cells})}

    rows = []
    prog = Progress(len(cells), label="cells", exp_id=EXP_ID, every=10)
    for opt, kind, K, ci, rep in cells:
        sp = spaces[ci]
        params = load_frozen(opt, scale.name)
        theta, r, obj, info = select_radius(
            sp, optimizer=opt, param_kind=kind, K=K, box_name=scale.box,
            folds=scale.folds, budget=scale.budget,
            seed=5000 + 31 * rep + 101 * ci, opt_params=params)
        rmse = sp.truth_error(r)
        rows.append(dict(
            exp_id=EXP_ID, optimizer=opt, method=LABELS[opt],
            is_control=opt in CONTROLS, param=kind, K=K, cycle=ci, repeat=rep,
            J=info["J"], rmse=rmse, radius_mean=info["radius_mean"],
            radius_std=info["radius_std"], n_evals=info["n_evals"],
            n_calls=info["n_calls"], n_cache_hits=info["n_cache_hits"],
            stalled=info["stalled"],
            rmse_oracle=oracle[ci]["error_opt"],
            penalty_pct=100.0 * (rmse / oracle[ci]["error_opt"] - 1.0)))
        prog.step(f"{LABELS[opt]:<22s} K={K} cycle={ci} J={info['J']:.4f}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    # Gap to the best value anyone found on that cycle and parameterization.
    df["J_best_cell"] = df.groupby(["K", "cycle"])["J"].transform("min")
    df["J_gap_pct"] = 100.0 * (df["J"] / df["J_best_cell"] - 1.0)

    summary = (df.groupby(["K", "method", "is_control"])
               .agg(J=("J", "mean"), J_gap_pct=("J_gap_pct", "mean"),
                    rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                    penalty_pct=("penalty_pct", "mean"),
                    evals=("n_evals", "mean"), n=("J", "count"))
               .reset_index().sort_values(["K", "J_gap_pct"]))
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, len(blocks), figsize=(5.4 * len(blocks), 3.8),
                             squeeze=False)
    for ax, (kind, K) in zip(axes[0], blocks):
        sub = df[df["K"] == K]
        order = (sub.groupby("method")["J_gap_pct"].mean()
                 .sort_values().index.tolist())
        data = [sub[sub["method"] == m]["J_gap_pct"].values for m in order]
        bp = ax.boxplot(data, labels=order, vert=False, patch_artist=True,
                        widths=0.6)
        for patch, m in zip(bp["boxes"], order):
            is_ctrl = bool(sub[sub["method"] == m]["is_control"].iloc[0])
            patch.set_facecolor("lightgrey" if is_ctrl else "tab:blue")
            patch.set_alpha(0.7)
        ax.set_xlabel("gap to the best $J$ found on the cycle (%)")
        ax.set_title(f"{kind}, K = {K} free radii\n(grey: controls)")
    fig.suptitle(f"EXP-04: equal budget of {scale.budget} evaluations of $J$")
    fig.tight_layout()
    ctx.save_fig(fig, "metaheuristics.png")

    ctx.save_latex(
        latex_table(summary.drop(columns=["is_control", "rmse_sd"]),
                    "Searches at equal budget on the tapered parameterization.",
                    "tab:meta"),
        "metaheuristics.tex")

    ctx.finish(summary=dict(n_cells=len(df)))


if __name__ == "__main__":
    main(parse_cli())
