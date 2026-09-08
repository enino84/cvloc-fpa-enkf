# -*- coding: utf-8 -*-
"""
EXP-02-BASELINES

The reference rows of the main table.

Six configurations, all on the same cycles, so that every number in Table 1 is
comparable line by line:

    background, no assimilation          what the ensemble knows on its own
    no localization                      the raw sample covariance, unfiltered
    fixed radius, tuned once             the operational default the method
                                         is supposed to replace
    uniform radius, cross-validation     the method
    uniform radius, truth-based optimum  the oracle, not admissible
    one radius per variable, CV          the over-parameterized version

The gap between the third and fourth rows is the practical case for the method;
the gap between the fourth and fifth is what it costs to be honest about not
having the truth; the gap between the fourth and sixth is the negative result
that EXP-07 then takes apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvloc.cycles import select_radius, truth_optimum
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    load_frozen, method_allowed, parse_cli, Progress,
                    shard_cells)

EXP_ID = "EXP-02-BASELINES"
DESCRIPTION = ("Reference configurations on a climatological ensemble: "
               "background, no localization, fixed, CV, oracle, per-variable.")

METHODS = [
    "background, no assimilation",
    "no localization",
    "fixed radius r = 2",
    "uniform radius, cross-validation",
    "uniform radius, truth-based optimum",
    "40 free radii, cross-validation",
]


def main(scale_name=None):
    scale = get_scale(scale_name)
    ctx = ExperimentContext(EXP_ID, DESCRIPTION, scale)
    params = load_frozen("fpa", scale.name)

    built = {ratio: build_spaces(scale, obs_ratio=ratio)
             for ratio in scale.obs_ratios}
    cells = shard_cells([(m, ratio, ci) for m in METHODS
                         for ratio in scale.obs_ratios
                         for ci in range(scale.n_cycles)
                         if method_allowed(m)])

    rows = []
    prog = Progress(len(cells), label="cells", exp_id=EXP_ID, every=10)
    for label, ratio, ci in cells:
        cycles, spaces = built[ratio]
        sp, cyc = spaces[ci], cycles[ci]
        seed = 3000 + 97 * ci
        radius = np.nan
        J = np.nan

        if label.startswith("background"):
            rmse = cyc.background_rmse()
        elif label.startswith("no localization"):
            # A radius far beyond the domain leaves the taper at one
            # everywhere, which is exactly the unlocalized filter.
            rmse = sp.rmse(sp.analysis_mean(None))
        elif label.startswith("fixed"):
            radius = 2.0
            rmse = sp.truth_error(np.full(sp.n, radius))
        elif label.startswith("uniform radius, cross"):
            theta, r, obj, info = select_radius(
                sp, optimizer="fpa", param_kind="uniform", K=1,
                box_name=scale.box, folds=scale.folds, budget=scale.budget,
                seed=seed, opt_params=params)
            radius, J, rmse = float(r[0]), info["J"], sp.truth_error(r)
        elif label.startswith("uniform radius, truth"):
            to = truth_optimum(sp, scale.sweep_lo, scale.sweep_hi,
                               scale.sweep_points)
            radius, rmse = to["r_opt"], to["error_opt"]
        else:
            theta, r, obj, info = select_radius(
                sp, optimizer="fpa", param_kind="per-variable", K=sp.n,
                box_name=scale.box, folds=scale.folds, budget=scale.budget,
                seed=seed, opt_params=params)
            radius, J, rmse = float(np.mean(r)), info["J"], sp.truth_error(r)

        rows.append(dict(exp_id=EXP_ID, method=label, obs_ratio=ratio,
                         p_obs=int(sp.p), cycle=ci,
                         seed=cyc.meta["seed"], rmse=rmse, radius=radius, J=J,
                         admissible=not label.startswith("uniform radius, truth")))
        prog.step(f"{label:<38s} p/n={ratio} cycle={ci} rmse={rmse:.4f}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    summary = (df.groupby(["obs_ratio", "method"])
               .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                    radius=("radius", "mean"), n=("rmse", "count"))
               .reset_index())
    order = {m: i for i, m in enumerate(METHODS)}
    summary = summary.sort_values(
        ["obs_ratio", "method"], key=lambda c: c.map(order) if c.name == "method" else c)
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    ctx.save_latex(
        latex_table(summary[["method", "rmse", "radius"]],
                    f"Single cycle, climatological ensemble, "
                    f"$N = {scale.ensemble_size}$, $n = p = {scale.n_state}$. "
                    f"Analysis RMSE, mean over {scale.n_cycles} independent cycles.",
                    "tab:baselines"),
        "baselines.tex")

    ctx.finish(summary=dict(n_cells=len(df),
                            best=summary.sort_values("rmse").iloc[0].to_dict()))


if __name__ == "__main__":
    main(parse_cli())
