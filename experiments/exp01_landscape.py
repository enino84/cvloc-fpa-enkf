# -*- coding: utf-8 -*-
"""
EXP-01-LANDSCAPE

Does the radius matter, and do the two curves agree?

A dense sweep of the uniform radius, scored twice: against the truth, which an
operational system could not do, and against the cross-validated criterion,
which it could. Everything the paper claims about the shape of the problem
comes from here.

Three quantities are reported.

* The **spread**: how much the analysis error varies across the admissible
  range, relative to its minimum. If that number is small the whole exercise is
  pointless, so it belongs at the front of the paper rather than in an
  appendix.
* The **asymmetry**: the error curve is flat towards short radii and rises
  steeply towards long ones, so underestimating the radius is much cheaper than
  overestimating it. This is why the truth-based arg-min is weakly identified
  and why the paper reports the error penalty rather than agreement between
  arg-minima.
* The **noise floor** of J: the criterion contains the observation error
  variance of the validation set, which does not depend on r. It does not move
  the minimizer but it flattens the landscape, which is what makes the search
  problem harder than the error curve suggests.

Also produces the covariance panels: raw sample covariance, the taper implied
by the selected radius, and the localized covariance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvloc.cycles import cv_sweep, truth_optimum
from cvloc.features import cluster
from cvloc.persist import SnapshotWriter
from cvloc.taper import taper_matrix
from common import (ExperimentContext, build_spaces, get_scale, latex_table,
                    parse_cli, setup_matplotlib, shard_cells)

EXP_ID = "EXP-01-LANDSCAPE"
DESCRIPTION = ("Dense sweep of the uniform radius against the truth-based "
               "error and the cross-validated cost, Lorenz 96.")


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    ctx = ExperimentContext(
        EXP_ID, DESCRIPTION, scale,
        extra_config=dict(sweep_points=scale.sweep_points,
                          sweep_range=[scale.sweep_lo, scale.sweep_hi]))

    built = {ratio: build_spaces(scale, obs_ratio=ratio)
             for ratio in scale.obs_ratios}
    cells = shard_cells([(ratio, ci) for ratio in scale.obs_ratios
                         for ci in range(scale.n_cycles)])

    snaps = SnapshotWriter()
    err_curves, cv_curves, rows = [], [], []
    for ratio, ci in cells:
        cycles, spaces = built[ratio]
        sp = spaces[ci]
        to = truth_optimum(sp, scale.sweep_lo, scale.sweep_hi,
                           scale.sweep_points)
        cs = cv_sweep(sp, folds=scale.folds, lo=scale.sweep_lo,
                      hi=scale.sweep_hi, n_points=scale.sweep_points)
        grid = to["grid"]

        # Store the cycle so every covariance panel can be redrawn, or redrawn
        # with a different radius, without regenerating the ensemble.
        r_sel = np.full(sp.n, cs["r_opt"])
        _, Xa = sp.analysis_ensemble(r_sel)
        snaps.add(tag=f"ratio{ratio}/cycle{ci}", Xb=cycles[ci].Xb, Xa=Xa,
                  x_true=cycles[ci].x_true, obs_idx=cycles[ci].obs_idx,
                  y=cycles[ci].y, r_inv=cycles[ci].r_inv, r=r_sel,
                  obs_ratio=ratio, cycle=ci, r_cv=cs["r_opt"],
                  r_truth=to["r_opt"], seed=cycles[ci].meta["seed"])

        # Width of the flat region: the interval of radii whose error is within
        # one percent of the minimum. A wide interval is what makes the
        # arg-min a bad summary of the curve.
        near = to["error"] <= 1.01 * to["error_opt"]
        flat_lo, flat_hi = float(grid[near].min()), float(grid[near].max())

        if ratio == scale.obs_ratios[0]:
            err_curves.append(to["error"])
            cv_curves.append(cs["cost"])
        rows.append(dict(
            exp_id=EXP_ID, obs_ratio=ratio, p_obs=int(sp.p),
            cycle=ci, seed=cycles[ci].meta["seed"],
            background_rmse=cycles[ci].background_rmse(),
            r_truth=to["r_opt"], err_truth=to["error_opt"],
            err_worst=to["error_worst"],
            spread_pct=100.0 * (to["error_worst"] / to["error_opt"] - 1.0),
            r_cv=cs["r_opt"], cost_cv=cs["cost_opt"],
            err_at_cv=float(sp.truth_error(np.full(sp.n, cs["r_opt"]))),
            flat_lo=flat_lo, flat_hi=flat_hi,
            cost_floor=float(cs["cost"].max() if cs["cost"][0] > cs["cost"][-1]
                             else cs["cost"][0]),
        ))

    df = pd.DataFrame(rows)
    df["penalty_pct"] = 100.0 * (df["err_at_cv"] / df["err_truth"] - 1.0)
    ctx.save_table(df, "metrics.csv")
    ctx.save_snapshots(snaps)

    curves = pd.DataFrame(dict(radius=grid,
                               err_mean=np.mean(err_curves, axis=0),
                               err_sd=np.std(err_curves, axis=0),
                               cv_mean=np.mean(cv_curves, axis=0),
                               cv_sd=np.std(cv_curves, axis=0)))
    ctx.save_table(curves, "curves.csv")

    if ctx.is_shard:
        ctx.finish(summary=dict(n_cycles=len(df)))
        return

    E, C = np.array(err_curves), np.array(cv_curves)
    me, mc = E.mean(0), C.mean(0)

    by_ratio = (df.groupby("obs_ratio")
                .agg(p_obs=("p_obs", "mean"), spread=("spread_pct", "mean"),
                     r_truth=("r_truth", "mean"), r_cv=("r_cv", "mean"),
                     err_truth=("err_truth", "mean"),
                     penalty=("penalty_pct", "mean"),
                     flat_width=("flat_hi", "mean")).reset_index())
    ctx.save_table(by_ratio, "by_ratio.csv")
    print()
    print(by_ratio.to_string(index=False))

    df = df[df["obs_ratio"] == scale.obs_ratios[0]]
    summary = pd.DataFrame([dict(
        n_cycles=len(df),
        mean_spread_pct=float(df["spread_pct"].mean()),
        mean_r_truth=float(df["r_truth"].mean()),
        mean_r_cv=float(df["r_cv"].mean()),
        mean_penalty_pct=float(df["penalty_pct"].mean()),
        median_penalty_pct=float(df["penalty_pct"].median()),
        pooled_r_truth=float(grid[int(me.argmin())]),
        pooled_r_cv=float(grid[int(mc.argmin())]),
        mean_flat_width=float((df["flat_hi"] - df["flat_lo"]).mean()),
        rank_correlation=float(pd.Series(me).corr(pd.Series(mc),
                                                  method="spearman")),
    )])
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.T.to_string())

    # ---------------- figure: the two curves ----------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(grid, me, "ko-", ms=3, label="analysis RMSE (truth)")
    ax.fill_between(grid, me - E.std(0), me + E.std(0), color="k", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("uniform radius $r$")
    ax.set_ylabel("analysis RMSE")
    ax2 = ax.twinx()
    ax2.plot(grid, mc, "s--", color="tab:blue", ms=3, label="CV cost")
    ax2.fill_between(grid, mc - C.std(0), mc + C.std(0),
                     color="tab:blue", alpha=0.12)
    ax2.set_ylabel("CV cost", color="tab:blue")
    ax2.grid(False)
    ax.axvline(grid[int(mc.argmin())], color="tab:blue", ls=":", lw=1)
    ax.set_title("both curves are minimised in the same region")
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ctx.save_fig(fig, "sweep_truth_vs_cv.png")

    # ---------------- figure: covariance panels ----------------
    cycles, spaces = built[scale.obs_ratios[0]]
    sp = spaces[0]
    r_star = float(df["r_cv"].iloc[0])
    B = sp.sample_covariance()
    C_taper = taper_matrix(sp.n, np.full(sp.n, r_star))
    labels, feats, cinfo = cluster(cycles[0].Xb, K=None, K_range=(2, 6))

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    v = np.abs(B).max()
    im = axes[0, 0].imshow(B, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[0, 0].set_title(f"(a) raw sample covariance\nN = {sp.N}, climatological")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    im = axes[0, 1].imshow(C_taper, cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title(f"(b) Gaspari-Cohn taper, $r$ = {r_star:.2f}")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    im = axes[0, 2].imshow(B * C_taper, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[0, 2].set_title(r"(c) localized covariance $B \circ C$")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].step(np.arange(sp.n), np.full(sp.n, r_star), where="mid",
                    color="tab:blue", label=f"uniform, CV = {r_star:.2f}")
    axes[1, 0].axhline(float(df["r_truth"].iloc[0]), color="k", ls=":",
                       label=f"uniform, oracle = {df['r_truth'].iloc[0]:.2f}")
    axes[1, 0].set_xlabel("grid point $j$")
    axes[1, 0].set_ylabel("radius $r_j$")
    axes[1, 0].set_title("(d) selected radius")
    axes[1, 0].legend(fontsize=7)

    sc = axes[1, 1].scatter(feats[:, 0], feats[:, 1], c=labels, cmap="Spectral",
                            edgecolor="k", linewidth=0.4, s=28)
    axes[1, 1].set_xlabel(r"background variance $\sigma_j^2$")
    axes[1, 1].set_ylabel(r"decay length $L_j$")
    axes[1, 1].set_title(f"(e) feature space, {cinfo['K']} clusters by silhouette")

    axes[1, 2].plot(grid, E[0], "ko-", ms=3, label="RMSE (truth)")
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_xlabel("uniform radius $r$")
    axes[1, 2].set_ylabel("analysis RMSE")
    ax2 = axes[1, 2].twinx()
    ax2.plot(grid, C[0], "s--", color="tab:blue", ms=3)
    ax2.set_ylabel("CV cost", color="tab:blue")
    ax2.grid(False)
    axes[1, 2].axvline(grid[int(C[0].argmin())], color="tab:blue", ls="--", lw=1)
    axes[1, 2].set_title("(f) one cycle: truth vs CV")
    axes[1, 2].legend(fontsize=7)

    fig.suptitle("Estimated localization radius and the resulting covariance "
                 "structure - one cycle, Lorenz-96")
    fig.tight_layout()
    ctx.save_fig(fig, "covariance_panels.png")

    ctx.save_latex(
        latex_table(summary.T.reset_index().rename(
            columns={"index": "quantity", 0: "value"}),
            "Shape of the localization landscape over independent cycles.",
            "tab:landscape"),
        "landscape.tex")

    ctx.finish(summary=summary.iloc[0].to_dict())


if __name__ == "__main__":
    main(parse_cli())
