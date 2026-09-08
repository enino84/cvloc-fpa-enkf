# -*- coding: utf-8 -*-
"""
EXP-07-KRADII

Why does the analysis degrade when the estimator is given more free radii?

The draft reports that the error grows monotonically with the number of free
radii K while the cross-validated cost attained at the optimum does not
improve, and reads this as a property of the parameterization: in a homogeneous
configuration all grid points are exchangeable, so there is nothing for a
spatially varying radius to exploit, and the estimation variance it introduces
is paid in full. Yang's reading is different: the step size may be too large,
the box unbounded, the population too small and the run too short, so the
search simply fails to find the better optimum that is there.

Both readings predict the same headline number, so the draft as it stands
cannot distinguish them. This experiment is built to.

The key observation is that the parameterizations are **nested**. A uniform
radius is a K-block radius with all blocks equal, so the attainable minimum of
J is non-increasing in K by construction. Therefore

* if the attained J *rises* with K, the search failed, full stop: the optimum
  found at K = 1 was available at every larger K and was not found. This is
  Yang's hypothesis, and it is falsifiable.
* if the attained J *falls* with K while the true error rises, the criterion is
  being overfitted: the extra freedom is being used, but to fit the validation
  observations rather than the state.
* if neither moves, the search is at the boundary of what the budget allows and
  the freedom is neither exploited nor harmful in J, which is the reading of
  the draft.

Four factors are crossed so that each of Yang's three suggestions is switched
on and off independently: the number of free radii, whether the budget is held
fixed or scaled with K, the admissible box, and whether the best radius is
broadcast across components. The verdict column applies the test above to every
combination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvloc.cycles import select_radius
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    load_frozen, parse_cli, Progress, setup_matplotlib,
                    shard_cells)

EXP_ID = "EXP-07-KRADII"
DESCRIPTION = ("Ablation of the number of free radii against budget, bounds "
               "and the broadcast move: parameterization or search?")


def verdict(J_ratio, rmse_ratio, tol=0.005):
    """Apply the nesting test to one (K, configuration) cell."""
    if J_ratio > 1.0 + tol:
        return "search fails"
    if J_ratio < 1.0 - tol and rmse_ratio > 1.0 + tol:
        return "overfitting"
    if abs(J_ratio - 1.0) <= tol and rmse_ratio > 1.0 + tol:
        return "freedom not exploited"
    if rmse_ratio <= 1.0 + tol:
        return "freedom pays"
    return "inconclusive"


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(
        EXP_ID, DESCRIPTION, scale,
        extra_config=dict(K_list=list(scale.K_list), boxes=list(scale.boxes)))

    cycles, spaces = build_spaces(scale)
    base = load_frozen("fpa", scale.name)

    factors = [
        # (label, budget rule, box, broadcast)
        ("fixed budget, wide box", "fixed", "wide", False),
        ("fixed budget, informed box", "fixed", "informed", False),
        ("scaled budget, informed box", "scaled", "informed", False),
        ("scaled budget, informed box, broadcast", "scaled", "informed", True),
    ]

    cells = shard_cells([(f, K, ci, rep)
                         for f in factors
                         for K in scale.K_list
                         for ci in range(len(spaces))
                         for rep in range(scale.n_repeats)])

    rows = []
    prog = Progress(len(cells), label="cells", exp_id=EXP_ID, every=20)
    for (flabel, brule, boxname, bcast), K, ci, rep in cells:
        sp = spaces[ci]
        # The scaled budget is capped. Without a ceiling the K = 40 cells cost
        # more than the rest of the suite combined, and the question here is
        # whether *some* extra budget rescues the parameterization, not how
        # much budget it would take to brute-force it.
        scale_factor = min(float(scale.budget_scale_cap), max(1.0, K / 4.0))
        budget = (scale.budget if brule == "fixed"
                  else int(scale.budget * scale_factor))
        params = {**base, "broadcast": bool(bcast)}
        kind = "uniform" if K == 1 else "blocks"
        theta, r, obj, info = select_radius(
            sp, optimizer="fpa", param_kind=kind, K=K, box_name=boxname,
            folds=scale.folds, budget=budget,
            seed=9000 + 31 * rep + 101 * ci, opt_params=params)
        rows.append(dict(
            exp_id=EXP_ID, factor=flabel, budget_rule=brule, box=boxname,
            broadcast=bool(bcast), K=K, cycle=ci, repeat=rep, budget=budget,
            J=info["J"], rmse=sp.truth_error(r),
            radius_mean=info["radius_mean"], radius_std=info["radius_std"],
            n_evals=info["n_evals"],
            theta=str(np.round(theta, 3).tolist())))
        prog.step(f"{flabel:<38s} K={K:<3d} cycle={ci}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    # Normalize against K = 1 within the same factor and cycle: that is the
    # nested comparison the verdict rests on.
    ref = (df[df["K"] == 1].groupby(["factor", "cycle"])[["J", "rmse"]]
           .mean().rename(columns={"J": "J_ref", "rmse": "rmse_ref"}))
    df = df.join(ref, on=["factor", "cycle"])
    df["J_ratio"] = df["J"] / df["J_ref"]
    df["rmse_ratio"] = df["rmse"] / df["rmse_ref"]

    summary = (df.groupby(["factor", "K"])
               .agg(J_ratio=("J_ratio", "mean"), rmse_ratio=("rmse_ratio", "mean"),
                    J=("J", "mean"), rmse=("rmse", "mean"),
                    radius_std=("radius_std", "mean"),
                    budget=("budget", "mean"), n=("J", "count"))
               .reset_index())
    summary["verdict"] = [verdict(j, e) for j, e in
                          zip(summary["J_ratio"], summary["rmse_ratio"])]
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for flabel, grp in summary.groupby("factor"):
        g = grp.sort_values("K")
        axes[0].plot(g["K"], g["rmse_ratio"], "o-", label=flabel)
        axes[1].plot(g["K"], g["J_ratio"], "s-", label=flabel)
    axes[0].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("number of free radii $K$")
    axes[0].set_ylabel(r"RMSE relative to $K=1$")
    axes[0].set_title("(a) does the extra freedom cost accuracy?")
    axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of free radii $K$")
    axes[1].set_ylabel(r"attained $J$ relative to $K=1$")
    axes[1].set_title("(b) above 1 means the search failed:\n"
                      "$K=1$ is nested inside every larger $K$")
    axes[1].legend(fontsize=7)

    # The estimated profile in state space, for the largest K.
    K_max = int(df["K"].max())
    prof = df[(df["K"] == K_max) & (df["cycle"] == 0) &
              (df["factor"] == factors[-1][0])]
    if not prof.empty:
        theta = np.array(eval(prof.iloc[0]["theta"]))
        n = spaces[0].n
        assign = (np.arange(n) * K_max) // n
        axes[2].step(np.arange(n), theta[assign], where="mid",
                     color="grey", label=f"K = {K_max}")
    for K in [k for k in (1, 4) if k in set(df["K"])]:
        sub = df[(df["K"] == K) & (df["cycle"] == 0) &
                 (df["factor"] == factors[-1][0])]
        if sub.empty:
            continue
        th = np.array(eval(sub.iloc[0]["theta"]))
        n = spaces[0].n
        assign = (np.arange(n) * K) // n
        axes[2].step(np.arange(n), th[assign], where="mid", label=f"K = {K}")
    axes[2].set_xlabel("grid point $j$")
    axes[2].set_ylabel("estimated radius $r_j$")
    axes[2].set_title("(c) estimated profiles")
    axes[2].legend(fontsize=7)

    fig.suptitle("EXP-07: is it the parameterization or the search?")
    fig.tight_layout()
    ctx.save_fig(fig, "k_radii.png")

    ctx.save_latex(
        latex_table(summary[["factor", "K", "J_ratio", "rmse_ratio", "verdict"]],
                    "Number of free radii against budget, bounds and the "
                    "broadcast move. $J$ above one at $K>1$ implies a search "
                    "failure, since the $K=1$ solution is admissible at every "
                    "larger $K$.",
                    "tab:kradii"),
        "kradii.tex")

    verdicts = summary[summary["K"] > 1]["verdict"].value_counts().to_dict()
    ctx.finish(summary=dict(n_cells=len(df), verdicts=verdicts))


if __name__ == "__main__":
    main(parse_cli())
