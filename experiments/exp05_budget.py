# -*- coding: utf-8 -*-
"""
EXP-05-BUDGET

How much search does the criterion actually need, and does more help?

Two axes, crossed: the number of unique objective evaluations, and the
population size. Both are Yang's questions. Longer runs are the direct test of
whether the multi-radius result of the draft is a budget artefact, and the
population sweep goes up to sizes far larger than the ten the draft uses,
because a population of ten in a twenty-dimensional box is a very small sample
of it.

Two things are reported that a plain "final cost against budget" curve hides.

* The **convergence trace**: the running minimum of J against the evaluation
  count, averaged over cycles. If the trace is still descending when the budget
  ends, the budget is the binding constraint; if it flattened long before, it
  is not, and the answer to "run it longer" is that it will not help.

* The **budget at which the trace stops improving**, defined as the first
  evaluation after which the running minimum falls by less than one part in a
  thousand. That is the number the paper should quote when it says how
  expensive the method is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvloc.cycles import select_radius
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    load_frozen, parse_cli, Progress, setup_matplotlib,
                    shard_cells)

EXP_ID = "EXP-05-BUDGET"
DESCRIPTION = ("Convergence of the search against the evaluation budget and "
               "the population size, for one and for several free radii.")


def plateau_at(trace, tol=1e-3):
    """First index after which the running minimum improves by less than tol."""
    t = np.asarray(trace, dtype=float)
    if t.size < 2:
        return int(t.size)
    rel = np.abs(np.diff(t)) / np.maximum(np.abs(t[:-1]), 1e-12)
    moving = np.flatnonzero(rel > tol)
    return int(moving[-1] + 2) if moving.size else 1


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(
        EXP_ID, DESCRIPTION, scale,
        extra_config=dict(budgets=list(scale.budgets),
                          pop_sizes=list(scale.pop_sizes)))

    cycles, spaces = build_spaces(scale)
    K_multi = max(scale.K_list)
    blocks = [("uniform", 1), ("blocks", K_multi)]
    max_budget = max(scale.budgets)

    cells = shard_cells([(kind, K, budget, pop, ci, rep)
                         for kind, K in blocks
                         for budget in scale.budgets
                         for pop in scale.pop_sizes
                         for ci in range(len(spaces))
                         for rep in range(scale.n_repeats)])

    rows, traces = [], {}
    prog = Progress(len(cells), label="cells", exp_id=EXP_ID, every=20)
    for kind, K, budget, pop, ci, rep in cells:
        sp = spaces[ci]
        params = {**load_frozen("fpa", scale.name), "pop_size": pop}
        theta, r, obj, info = select_radius(
            sp, optimizer="fpa", param_kind=kind, K=K, box_name=scale.box,
            folds=scale.folds, budget=budget,
            seed=6000 + 31 * rep + 101 * ci, opt_params=params)
        trace = obj.best_trace()
        rows.append(dict(
            exp_id=EXP_ID, param=kind, K=K, budget=budget, pop_size=pop,
            cycle=ci, repeat=rep, J=info["J"], rmse=sp.truth_error(r),
            n_evals=info["n_evals"], radius_mean=info["radius_mean"],
            plateau=plateau_at(trace), stalled=info["stalled"]))
        if budget == max_budget:
            traces.setdefault((K, pop), []).append(
                np.pad(trace, (0, max_budget - trace.size),
                       mode="edge") if trace.size else np.full(max_budget, np.nan))
        prog.step(f"K={K} budget={budget} S={pop} cycle={ci}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    df["J_best_cell"] = df.groupby(["K", "cycle"])["J"].transform("min")
    df["J_gap_pct"] = 100.0 * (df["J"] / df["J_best_cell"] - 1.0)

    summary = (df.groupby(["K", "budget", "pop_size"])
               .agg(J=("J", "mean"), J_gap_pct=("J_gap_pct", "mean"),
                    rmse=("rmse", "mean"), plateau=("plateau", "mean"),
                    n=("J", "count")).reset_index())
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for K in sorted(df["K"].unique()):
        g = df[df["K"] == K].groupby("budget")["rmse"].mean()
        axes[0].plot(g.index, g.values, "o-", label=f"K = {K}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("budget (evaluations of $J$)")
    axes[0].set_ylabel("analysis RMSE")
    axes[0].set_title("(a) does more search buy accuracy?")
    axes[0].legend()

    for K in sorted(df["K"].unique()):
        g = df[df["K"] == K].groupby("pop_size")["J_gap_pct"].mean()
        axes[1].plot(g.index, g.values, "s-", label=f"K = {K}")
    axes[1].set_xlabel("population size $S$")
    axes[1].set_ylabel("gap to the best $J$ (%)")
    axes[1].set_title("(b) does a larger population help?")
    axes[1].legend()

    for (K, pop), tr in sorted(traces.items()):
        if pop != min(scale.pop_sizes):
            continue
        M = np.nanmean(np.array(tr), axis=0)
        axes[2].plot(np.arange(1, M.size + 1), M / M[-1], label=f"K = {K}")
    axes[2].set_xlabel("evaluations of $J$")
    axes[2].set_ylabel("running minimum, normalised")
    axes[2].set_title(f"(c) convergence trace, $S$ = {min(scale.pop_sizes)}")
    axes[2].legend()

    fig.suptitle("EXP-05: how much search the criterion needs")
    fig.tight_layout()
    ctx.save_fig(fig, "budget.png")

    ctx.save_latex(
        latex_table(summary, "Effect of the evaluation budget and the "
                             "population size.", "tab:budget"),
        "budget.tex")

    ctx.finish(summary=dict(n_cells=len(df),
                            mean_plateau=float(df["plateau"].mean())))


if __name__ == "__main__":
    main(parse_cli())
