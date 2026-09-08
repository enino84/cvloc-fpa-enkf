# -*- coding: utf-8 -*-
"""
EXP-08-HETERO

Where a spatially varying radius should actually pay.

The negative result of the draft is confined to a configuration in which every
grid point is statistically exchangeable: uniform grid, uniform forcing, every
point observed with the same error. There is then no reason for the radius to
vary in space, and the freedom to let it vary buys nothing while costing
estimation variance. The claim only becomes interesting if the converse holds,
so this experiment removes the exchangeability and checks.

Three ways of breaking it, run separately so the effect can be attributed:

``forcing``   the forcing varies across the domain, so the correlation length
              does too and no single radius is right everywhere.
``network``   the observation density alternates between dense and sparse
              regions. Where observations are scarce the analysis must reach
              further, which is the textbook reason for a spatially varying
              radius.
``both``      the two together, which is the realistic case.

Four parameterizations are compared on each: one radius, K contiguous blocks,
one radius per variable, and one radius per **cluster**, where clusters come
from ensemble-derived features (background variance and correlation decay
length) rather than from position. The clustered version is the one the draft's
discussion argues for: it lets the radius vary in space while keeping the
number of estimated parameters small, and it is admissible, since the features
are computed from the forecast ensemble alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvloc.cycles import select_radius, truth_optimum
from cvloc.features import cluster
from cvloc.persist import SnapshotWriter
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    load_frozen, parse_cli, Progress, setup_matplotlib,
                    shard_cells)

EXP_ID = "EXP-08-HETERO"
DESCRIPTION = ("Heterogeneous forcing and observation density: does a "
               "spatially varying radius pay when grid points are no longer "
               "exchangeable?")

REGIMES = {
    "homogeneous": dict(forcing="uniform", network="full"),
    "forcing": dict(forcing="blocks", network="full"),
    "network": dict(forcing="uniform", network="patchy"),
    "both": dict(forcing="blocks", network="patchy"),
}


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(EXP_ID, DESCRIPTION, scale,
                            extra_config=dict(regimes=list(REGIMES)))
    params = load_frozen("fpa", scale.name)

    K_blocks = max(2, min(4, max(scale.K_list)))
    built = {name: build_spaces(scale, forcing=cfg["forcing"],
                                network=cfg["network"])
             for name, cfg in REGIMES.items()}

    schemes = ["uniform", f"blocks K={K_blocks}", "clusters", "per-variable"]
    cells = shard_cells([(reg, sch, ci)
                         for reg in REGIMES
                         for sch in schemes
                         for ci in range(scale.n_cycles)])

    rows = []
    snaps = SnapshotWriter()
    prog = Progress(len(cells), label="cells", exp_id=EXP_ID, every=10)
    for reg, sch, ci in cells:
        cycles, spaces = built[reg]
        sp, cyc = spaces[ci], cycles[ci]

        labels, K, kind = None, 1, "uniform"
        if sch.startswith("blocks"):
            kind, K = "blocks", K_blocks
        elif sch == "clusters":
            labels, feats, cinfo = cluster(cyc.Xb, K=None, K_range=(2, 6))
            kind, K = "clusters", int(cinfo["K"])
        elif sch == "per-variable":
            kind, K = "per-variable", sp.n

        theta, r, obj, info = select_radius(
            sp, optimizer="fpa", param_kind=kind, K=K, labels=labels,
            box_name=scale.box, folds=scale.folds,
            budget=int(scale.budget * min(float(scale.budget_scale_cap),
                                          max(1.0, K / 4.0))),
            seed=11000 + 101 * ci, opt_params=params)

        to = truth_optimum(sp, scale.sweep_lo, scale.sweep_hi, scale.sweep_points)
        rmse = sp.truth_error(r)
        rows.append(dict(
            exp_id=EXP_ID, regime=reg, scheme=sch, K=int(K), cycle=ci,
            p_obs=int(cyc.p), J=info["J"], rmse=rmse,
            rmse_uniform_oracle=to["error_opt"],
            penalty_pct=100.0 * (rmse / to["error_opt"] - 1.0),
            radius_mean=info["radius_mean"], radius_std=info["radius_std"],
            profile=str(np.round(r, 3).tolist())))
        if ci == 0:
            # One cycle per (regime, scheme): enough to draw the estimated
            # radius profile against the forcing field it is supposed to track.
            _, Xa = sp.analysis_ensemble(r)
            snaps.add(tag=f"{reg}/{sch}/cycle{ci}", Xb=cyc.Xb, Xa=Xa,
                      x_true=cyc.x_true, obs_idx=cyc.obs_idx, y=cyc.y,
                      r_inv=cyc.r_inv, r=r, regime=reg, scheme=sch, cycle=ci,
                      K=int(K), rmse=rmse,
                      forcing=str(np.round(cyc.model.F_field, 3).tolist()))
        prog.step(f"{reg:<12s} {sch:<14s} cycle={ci} rmse={rmse:.4f}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    ctx.save_snapshots(snaps)
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    # Normalize against the uniform scheme within regime and cycle: the
    # question is not the absolute error, which differs between regimes, but
    # whether spatial freedom helps *within* a regime.
    ref = (df[df["scheme"] == "uniform"].groupby(["regime", "cycle"])["rmse"]
           .mean().rename("rmse_uniform"))
    df = df.join(ref, on=["regime", "cycle"])
    df["gain_pct"] = 100.0 * (1.0 - df["rmse"] / df["rmse_uniform"])

    summary = (df.groupby(["regime", "scheme"])
               .agg(K=("K", "mean"), rmse=("rmse", "mean"),
                    rmse_sd=("rmse", "std"), gain_pct=("gain_pct", "mean"),
                    J=("J", "mean"), radius_std=("radius_std", "mean"),
                    n=("rmse", "count")).reset_index())
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    regimes = [r for r in REGIMES if r in set(summary["regime"])]
    width = 0.8 / len(schemes)
    x = np.arange(len(regimes))
    for k, sch in enumerate(schemes):
        vals = [float(summary[(summary["regime"] == r) &
                              (summary["scheme"] == sch)]["gain_pct"].mean())
                if not summary[(summary["regime"] == r) &
                               (summary["scheme"] == sch)].empty else np.nan
                for r in regimes]
        axes[0].bar(x + k * width, vals, width, label=sch)
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_xticks(x + 0.4 - width / 2)
    axes[0].set_xticklabels(regimes)
    axes[0].set_ylabel("RMSE gain over a uniform radius (%)")
    axes[0].set_title("(a) spatial freedom pays only where\nthe domain is not exchangeable")
    axes[0].legend(fontsize=7)

    # The estimated profile in the hardest regime, against the forcing field.
    hard = df[(df["regime"] == "both") & (df["cycle"] == 0)]
    if not hard.empty:
        for sch in ["uniform", "clusters", "per-variable"]:
            sub = hard[hard["scheme"] == sch]
            if sub.empty:
                continue
            prof = np.array(eval(sub.iloc[0]["profile"]))
            axes[1].step(np.arange(prof.size), prof, where="mid", label=sch)
        cyc0 = built["both"][0][0]
        ax2 = axes[1].twinx()
        ax2.plot(cyc0.model.F_field, color="k", ls=":", lw=1, alpha=0.6)
        ax2.set_ylabel("forcing $F_j$", color="grey")
        ax2.grid(False)
    axes[1].set_xlabel("grid point $j$")
    axes[1].set_ylabel("estimated radius $r_j$")
    axes[1].set_title("(b) does the profile track the heterogeneity?")
    axes[1].legend(fontsize=7, loc="upper left")

    fig.suptitle("EXP-08: heterogeneous forcing and observation density")
    fig.tight_layout()
    ctx.save_fig(fig, "heterogeneous.png")

    ctx.save_latex(
        latex_table(summary[["regime", "scheme", "K", "rmse", "gain_pct"]],
                    "Spatially varying radius under heterogeneous forcing and "
                    "observation density.", "tab:hetero"),
        "hetero.tex")

    ctx.finish(summary=dict(n_cells=len(df)))


if __name__ == "__main__":
    main(parse_cli())
