# -*- coding: utf-8 -*-
"""
EXP-00-TUNING

Are the search hyperparameters calibrated fairly?

Every later experiment reports a comparison between searches. That comparison
is meaningless if one of them was tuned and the others were not, and it is
worse than meaningless if the tuning looked at the truth. So the parameters are
calibrated here, on **held-out cycles disjoint from every other experiment**,
scored by the cross-validated objective alone.

For FPA this also answers Yang's first point directly. The global step is

    dS = gamma * L(lambda) * (x - x_best)

and the reference implementation folds a factor of 0.01 into L. An
implementation that omits that factor and uses gamma = 1 takes steps a hundred
times too large, and late in the run the population never settles. The sweep
over gamma in {0.01 ... 1.0} at fixed everything else measures how much that
matters here rather than assuming it, and the winner is what the rest of the
suite uses.

Output ``results/frozen_params.json`` is read by every other experiment.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from cvloc.cycles import select_radius
from cvloc.metaheuristics import (CONTROLS, LABELS, METAHEURISTICS,
                                  defaults_for)
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    log, method_allowed, parse_cli, save_frozen, setup_matplotlib,
                    shard_cells)

EXP_ID = "EXP-00-TUNING"
DESCRIPTION = ("Hyperparameter calibration against the cross-validated "
               "objective on held-out cycles. No truth is consulted.")


def fpa_grid(scale):
    """The FPA configurations that are compared."""
    for gamma, pop, boxname in itertools.product(scale.gammas, scale.pop_sizes,
                                                 scale.boxes):
        yield dict(optimizer="fpa", box=boxname,
                   params=dict(gamma=gamma, pop_size=pop, switch_p=0.8,
                               lam=1.5, levy_scale=0.01),
                   label=f"fpa g={gamma} S={pop} box={boxname}")


def other_grid(scale):
    """A small grid for each competitor, so none of them is left untuned."""
    for pop in scale.pop_sizes[:2]:
        yield dict(optimizer="pso", box=scale.box,
                   params=dict(pop_size=pop, w=0.72, c1=1.49, c2=1.49),
                   label=f"pso S={pop}")
        yield dict(optimizer="de", box=scale.box,
                   params=dict(pop_size=pop, F=0.6, CR=0.9),
                   label=f"de S={pop}")
        yield dict(optimizer="ga", box=scale.box,
                   params=dict(pop_size=pop, p_mut=0.2),
                   label=f"ga S={pop}")
        yield dict(optimizer="firefly", box=scale.box,
                   params=dict(pop_size=pop, zeta=0.2),
                   label=f"firefly S={pop}")
    for cooling in (0.85, 0.92, 0.97):
        yield dict(optimizer="sa", box=scale.box,
                   params=dict(cooling=cooling, step_frac=0.15),
                   label=f"sa cool={cooling}")


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(
        EXP_ID, DESCRIPTION, scale,
        extra_config=dict(tuning_seeds=scale.tuning_seeds,
                          gammas=list(scale.gammas),
                          pop_sizes=list(scale.pop_sizes),
                          boxes=list(scale.boxes)))

    # Held-out cycles. Disjoint from scale.cycle_seeds by construction.
    _, spaces = build_spaces(scale, seeds=scale.tuning_seeds)

    configs = list(fpa_grid(scale)) + list(other_grid(scale))
    # Both K = 1 and a multi-radius case: a step size that works on one free
    # radius need not work on twenty, which is exactly the regime where the
    # negative result of the draft appears.
    K_values = [1, max(K for K in scale.K_list if K <= 8)]
    cells = [(cfg, K, ci, rep)
             for cfg in configs
             for K in K_values
             for ci in range(len(spaces))
             for rep in range(scale.n_repeats)
             if method_allowed(cfg["label"])]
    cells = shard_cells(cells)
    log(f"{len(cells)} tuning cells", EXP_ID)

    rows = []
    for cfg, K, ci, rep in cells:
        kind = "uniform" if K == 1 else "blocks"
        theta, r, obj, info = select_radius(
            spaces[ci], optimizer=cfg["optimizer"], param_kind=kind, K=K,
            box_name=cfg["box"], folds=scale.folds, budget=scale.budget,
            seed=7000 + 31 * rep + 101 * ci, opt_params=cfg["params"])
        rows.append(dict(
            exp_id=EXP_ID, label=cfg["label"], optimizer=cfg["optimizer"],
            box=cfg["box"], K=K, cycle=ci, repeat=rep,
            J=info["J"], n_evals=info["n_evals"], n_calls=info["n_calls"],
            radius_mean=info["radius_mean"], radius_std=info["radius_std"],
            **{f"p_{k}": v for k, v in cfg["params"].items()}))

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")

    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    # The criterion for "best" is the mean attained J, normalized per cycle so
    # that cycles with a naturally larger cost do not dominate the average.
    df["J_rel"] = df.groupby(["K", "cycle"])["J"].transform(
        lambda s: s / s.min() if s.min() > 0 else s)
    summary = (df.groupby(["optimizer", "label", "K"])
               .agg(J_rel=("J_rel", "mean"), J=("J", "mean"),
                    J_sd=("J", "std"), evals=("n_evals", "mean"),
                    radius=("radius_mean", "mean"))
               .reset_index().sort_values(["K", "J_rel"]))
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.head(20).to_string(index=False))

    # Freeze the winner of each optimizer, judged on the multi-radius case,
    # since that is the harder of the two and the one the draft struggles with.
    K_hard = int(df["K"].max())
    frozen = {}
    for opt in sorted(df["optimizer"].unique()):
        sub = summary[(summary["optimizer"] == opt) & (summary["K"] == K_hard)]
        if sub.empty:
            sub = summary[summary["optimizer"] == opt]
        if sub.empty:
            continue
        best_label = sub.sort_values("J_rel").iloc[0]["label"]
        cfg = next(c for c in configs if c["label"] == best_label)
        frozen[opt] = {**defaults_for(opt), **cfg["params"]}
        log(f"frozen {opt}: {best_label}", EXP_ID)
    save_frozen(frozen, scale.name)

    # Figure: the gamma sweep, which is the question Yang raised.
    fpa = df[df["optimizer"] == "fpa"]
    if not fpa.empty and "p_gamma" in fpa.columns:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
        for ax, K in zip(axes, sorted(fpa["K"].unique())):
            sub = fpa[fpa["K"] == K]
            for pop, grp in sub.groupby("p_pop_size"):
                g = grp.groupby("p_gamma")["J_rel"].mean()
                ax.plot(g.index, g.values, "o-", label=f"S={pop}")
            ax.set_xscale("log")
            ax.set_xlabel(r"step size $\gamma$")
            ax.set_ylabel("attained $J$ / best of cycle")
            ax.set_title(f"K = {K} free radii")
            ax.legend()
        fig.suptitle("EXP-00: FPA step size against the cross-validated cost")
        ctx.save_fig(fig, "gamma_sweep.png")

    ctx.save_latex(
        latex_table(summary.head(12).drop(columns=["J_sd"]),
                    "Hyperparameter calibration against $J$ on held-out cycles.",
                    "tab:tuning"),
        "tuning.tex")

    ctx.finish(summary=dict(n_cells=len(df), frozen=frozen))


if __name__ == "__main__":
    main(parse_cli())
