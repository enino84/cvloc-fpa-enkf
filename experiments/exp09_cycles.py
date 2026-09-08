# -*- coding: utf-8 -*-
"""
EXP-09-CYCLES

The radius over a full filter run, and why the per-cycle estimate must not be
used raw.

This is the headline experiment. Everything before it is a single cycle from a
climatological ensemble, which is the right configuration for isolating what
the radius does but removes the question an operational centre actually has:
over a long run, with the network moving, does re-estimating the radius beat
choosing one and leaving it alone?

The finding this experiment is built around is that **the criterion identifies
the right radius and the raw per-cycle estimate still loses**. In a preliminary
run with half the network observed and relocated at random every cycle, a sweep
of fixed radii put the optimum at eight; the criterion selected 7.96 on
average, essentially exact; and it produced an analysis 38% worse than simply
holding the radius at eight. The reason is in the mean absolute change of the
radius between consecutive cycles, which was 7.67: the estimate averaged eight
while never being eight, swinging across the box from one cycle to the next.
The landscape is flat near its minimum, so the arg-min of one noisy realization
moves even when the quantity being estimated does not, and the analysis pays
for every swing.

The radius is therefore not a parameter to be estimated afresh each cycle. It
is a slowly varying quantity of which each cycle provides a noisy estimate, and
the experiment varies how those estimates are combined:

``none``     the raw estimate, which is what the draft describes
``window``   the geometric mean of the last few estimates
``ewma``     exponential smoothing, from nearly raw to nearly frozen
``freeze``   estimate for a few cycles, then hold the result fixed

against three references: a fixed radius a practitioner might guess, a fixed
radius tuned by a sweep against the truth, which is the oracle and is not
admissible, and the same machinery driven by a single-trajectory search.

Two further axes. Observation **density**, since a fully observed network gives
the radius almost nothing to do. And the observation **schedule**: ``fixed``
keeps the same stations throughout, ``random`` relocates them every cycle, as a
real network does. The second is the honest setting for the question, because
when the network moves no constant is right for every cycle.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

import cvloc  # noqa: F401  registers letkf-cv
from cvloc.model import Lorenz96CV
from cvloc.persist import SnapshotWriter
from common import (ExperimentContext, get_scale, latex_table, load_frozen,
                    method_allowed, network_for, parse_cli, Progress,
                    setup_matplotlib, shard_cells)
from pyteda.analysis.analysis_factory import AnalysisFactory
from pyteda.experiments import Scenario
from pyteda.observation import IsotropicDiagonal, LinearSelection
from pyteda.simulation import Simulation

EXP_ID = "EXP-09-CYCLES"
DESCRIPTION = ("Consecutive assimilation cycles with a moving observation "
               "network: how the per-cycle radius estimate must be combined "
               "over time.")

ORACLE_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)


def build_scenarios(scale, m_obs, schedule, verbose=False):
    """Frozen twin experiments, shared by every method at this cell.

    ``schedule='random'`` hands pyteda a fresh operator at every step, so the
    observed locations move each cycle. Each scenario has its own initial
    condition: sharing one reference trajectory would freeze the largest source
    of variability, which is where on the attractor the run happens to go.
    """
    model = Lorenz96CV(n=scale.n_state, F=scale.forcing, dt=scale.dt)
    n = model.get_number_of_variables()
    scenarios = []
    for s in scale.cycle_seeds[:scale.n_scenarios]:
        kick = np.random.default_rng(90_000 + s).normal(0.0, 1.0, size=n)
        x0 = model.propagate(model.get_initial_condition() + 0.5 * kick,
                             np.array([0.0, 10.0]))
        scenarios.append(Scenario.generate(
            model=model,
            operator_factory=lambda rng: LinearSelection(m=m_obs, n_state=n,
                                                         rng=rng),
            noise=IsotropicDiagonal(std=scale.obs_std, dim=m_obs),
            ensemble_size=scale.ensemble_size, x0_ref=x0,
            pert_xb=scale.pert_xb, spinup_xb=scale.spinup_xb,
            pert_ensemble=scale.pert_ensemble,
            spinup_ensemble=scale.spinup_ensemble,
            obs_freq=scale.obs_freq, end_time=scale.end_time, seed=s,
            operator_schedule=schedule))
    if verbose:
        print(f"  {len(scenarios)} scenarios, {scenarios[0].n_steps} cycles, "
              f"p={m_obs}/{n}, schedule={schedule}")
    return model, scenarios


def build_methods(scale):
    """Everything compared, with a flag for what is admissible.

    ``admissible`` marks whether an operational system could have produced the
    configuration. The oracle row could not: it needs a sweep against the true
    state. It is reported anyway, because it bounds what any selection rule
    could achieve and every other row is measured against it.
    """
    fpa = load_frozen("fpa", scale.name)
    sa = load_frozen("sa", scale.name)
    B = scale.budget
    return {
        "oracle: fixed radius by truth sweep":
            (dict(fixed_radius=None), False),
        "fixed radius r = 2":
            (dict(fixed_radius=2.0), True),
        "fixed radius r = 4":
            (dict(fixed_radius=4.0), True),
        "CV each cycle, raw":
            (dict(optimizer="fpa", warm_start=True, budget=B,
                  smoothing="none", opt_params=fpa), True),
        "CV each cycle, window of 5":
            (dict(optimizer="fpa", warm_start=True, budget=B,
                  smoothing="window", window=5, opt_params=fpa), True),
        "CV each cycle, window of 15":
            (dict(optimizer="fpa", warm_start=True, budget=B,
                  smoothing="window", window=15, opt_params=fpa), True),
        "CV each cycle, ewma 0.2":
            (dict(optimizer="fpa", warm_start=True, budget=B,
                  smoothing="ewma", alpha=0.2, opt_params=fpa), True),
        "CV each cycle, ewma 0.05":
            (dict(optimizer="fpa", warm_start=True, budget=B,
                  smoothing="ewma", alpha=0.05, opt_params=fpa), True),
        "CV then freeze after 5":
            (dict(optimizer="fpa", warm_start=True, budget=B,
                  smoothing="freeze", freeze_after=5, opt_params=fpa), True),
        "CV then freeze after 20":
            (dict(optimizer="fpa", warm_start=True, budget=B,
                  smoothing="freeze", freeze_after=20, opt_params=fpa), True),
        "CV + SA, ewma 0.05":
            (dict(optimizer="sa", warm_start=True, budget=B,
                  smoothing="ewma", alpha=0.05, opt_params=sa), True),
    }


def run_one(scenario, model, scale, cfg, seed, store_states_at=None):
    """One filter run.

    ``store_states_at`` is a sequence of fractions of the run at which pyteda
    keeps the forecast and analysis ensembles. Those are what a covariance
    panel needs and what no metrics table can carry: with a moving network the
    observed locations change every cycle, so neither the ensemble nor the
    network is reconstructible from a seed without replaying the scenario.
    Storing a handful of cycles per run rather than all of them keeps the
    archive to megabytes instead of hundreds of them.
    """
    analysis_cfg = dict(cfg)
    analysis_cfg.setdefault("model", model)
    analysis_cfg.setdefault("folds", scale.folds)
    analysis_cfg.setdefault("box_name", scale.box)
    analysis_cfg.setdefault("seed", seed)
    analysis = AnalysisFactory("letkf-cv", **analysis_cfg).create_analysis()

    t0 = time.perf_counter()
    failed, reason = False, ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation.from_scenario(
                scenario, analysis, inflation_factor=scale.inflation,
                method_rng=np.random.default_rng(2000 + seed),
                store_diagnostics=True, store_states_at=store_states_at)
            sim.run()
        err_a = np.asarray(sim.error_a, dtype=float)
        err_b = np.asarray(sim.error_b, dtype=float)

        # Trajectory tier: the analysis and background means at every cycle,
        # plus every per-cycle metric. Cheap enough to keep for every run, and
        # what any time series in the paper is drawn from.
        xb_mean, xa_mean = analysis.state_means
        xb_mean = np.asarray(xb_mean, dtype=float)
        xa_mean = np.asarray(xa_mean, dtype=float)
        truth = np.asarray(scenario.truth_trajectory,
                           dtype=float)[:xa_mean.shape[0]]
        traj = dict(
            xb_mean=xb_mean, xa_mean=xa_mean, x_true=truth,
            metrics=dict(
                # error_a and error_b from pyteda are *relative*; the absolute
                # RMSE is what the single-cycle experiments report, so both are
                # kept and neither has to be recomputed later from the other.
                rel_rmse_a=err_a, rel_rmse_b=err_b,
                rmse_a=np.sqrt(np.mean((xa_mean - truth) ** 2, axis=1)),
                rmse_b=np.sqrt(np.mean((xb_mean - truth) ** 2, axis=1)),
                spread_a=np.asarray(sim.spread_a, dtype=float),
                spread_b=np.asarray(sim.spread_b, dtype=float),
                crps_a=np.asarray(sim.crps_a, dtype=float),
                crps_b=np.asarray(sim.crps_b, dtype=float),
            ))
        snap = dict(Xb=sim.Xb_snapshots, Xa=sim.Xa_snapshots,
                    steps=np.asarray(sim.snapshot_steps),
                    times=np.asarray(sim.snapshot_times))
    except Exception as exc:
        failed, reason = True, f"{type(exc).__name__}: {exc}"
        err_a = err_b = np.full(scenario.n_steps, np.nan)
        snap, traj = None, None
    elapsed = time.perf_counter() - t0

    cut = int(np.ceil(scale.burn_in_frac * err_a.size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_a = float(np.nanmean(err_a[cut:]))
        mean_b = float(np.nanmean(err_b[cut:]))

    hist = getattr(analysis, "radius_history", [])
    r_used = np.array([h["radius_mean"] for h in hist]) if hist else np.array([])
    r_raw = (np.array([h.get("radius_raw", np.nan) for h in hist])
             if hist else np.array([]))
    ev = np.array([h["n_evals"] for h in hist]) if hist else np.array([])

    return dict(
        rel_rmse_a=mean_a, rel_rmse_b=mean_b, n_steps=int(err_a.size),
        burn_in=cut, elapsed_s=elapsed, failed=failed, fail_reason=reason,
        radius_used=float(np.mean(r_used)) if r_used.size else np.nan,
        radius_raw=float(np.nanmean(r_raw)) if r_raw.size else np.nan,
        drift_used=(float(np.mean(np.abs(np.diff(r_used))))
                    if r_used.size > 1 else 0.0),
        drift_raw=(float(np.nanmean(np.abs(np.diff(r_raw))))
                   if r_raw.size > 1 else 0.0),
        evals_per_cycle=float(ev.mean()) if ev.size else 0.0,
        spread_a=(float(np.nanmean(np.asarray(traj["metrics"]["spread_a"])[cut:]))
                  if traj else np.nan),
        crps_a=(float(np.nanmean(np.asarray(traj["metrics"]["crps_a"])[cut:]))
                if traj else np.nan),
        rmse_a_abs=(float(np.nanmean(np.asarray(traj["metrics"]["rmse_a"])[cut:]))
                    if traj else np.nan),
        rmse_b_abs=(float(np.nanmean(np.asarray(traj["metrics"]["rmse_b"])[cut:]))
                    if traj else np.nan),
    ), hist, err_a, snap, traj


def main(scale_name=None):
    scale = get_scale(scale_name)
    plt = setup_matplotlib()
    n_cycles_run = int(scale.end_time / scale.obs_freq)
    ctx = ExperimentContext(
        EXP_ID, DESCRIPTION, scale,
        extra_config=dict(obs_ratios=list(scale.obs_ratios),
                          schedules=list(scale.schedules),
                          n_cycles_per_run=n_cycles_run))

    methods = build_methods(scale)

    # A fully observed network cannot be relocated in any meaningful sense, so
    # that combination is dropped rather than run twice under two names.
    regimes = [(ratio, sched) for ratio in scale.obs_ratios
               for sched in scale.schedules
               if not (ratio >= 1.0 and sched == "random")]

    cells = shard_cells([(ratio, sched, label, sid)
                         for ratio, sched in regimes
                         for label in methods
                         for sid in range(scale.n_scenarios)
                         if method_allowed(label)])

    built, oracle_r = {}, {}
    snaps = SnapshotWriter()
    rows, hist_rows, curves = [], [], {}
    prog = Progress(len(cells), label="runs", exp_id=EXP_ID, every=2,
                    heartbeat_s=60.0)

    for ratio, sched, label, sid in cells:
        key = (ratio, sched)
        if key not in built:
            _, dense = network_for(ratio)
            m_obs = max(2, int(round(dense * scale.n_state)))
            built[key] = build_scenarios(scale, m_obs, sched, verbose=True)
        model, scenarios = built[key]

        cfg, admissible = methods[label]
        cfg = dict(cfg)

        if label.startswith("oracle"):
            # The sweep is run per scenario, not once per regime. Sweeping on
            # one scenario and applying the winner to the others makes the
            # "oracle" a radius tuned on somebody else's trajectory, which is
            # not an upper bound at all: an admissible rule can then beat it,
            # and a negative gap appears that means nothing. Each scenario gets
            # the radius that is actually best for it.
            okey = (ratio, sched, sid)
            if okey not in oracle_r:
                best = (None, np.inf)
                for r in ORACLE_GRID:
                    out, _, _, _, _ = run_one(scenarios[sid], model, scale,
                                              dict(fixed_radius=float(r)),
                                              1000 + 7 * sid)
                    if out["rel_rmse_a"] < best[1]:
                        best = (r, out["rel_rmse_a"])
                oracle_r[okey] = best[0]
                prog.note(f"oracle sweep p/n={ratio} {sched} s={sid}: "
                          f"r={best[0]} rmse={best[1]:.4f}")
            cfg["fixed_radius"] = float(oracle_r[okey])

        # Only a few methods get their ensembles kept: the raw estimate, the
        # smoothed one and the oracle are what a figure would compare, and
        # storing every method at every regime would multiply the archive by
        # eleven for no extra picture.
        keep = (label in ("CV each cycle, raw", "CV each cycle, ewma 0.05",
                          "CV then freeze after 20")
                or label.startswith("oracle")) and sid == 0
        out, hist, curve, snap, traj = run_one(
            scenarios[sid], model, scale, cfg, 1000 + 7 * sid,
            store_states_at=(scale.store_states_at if keep else None))
        rows.append(dict(exp_id=EXP_ID, method=label, admissible=admissible,
                         obs_ratio=ratio, schedule=sched, scenario_id=sid,
                         budget=cfg.get("budget", 0),
                         smoothing=cfg.get("smoothing", "n/a"), **out))
        curves.setdefault((ratio, sched, label), []).append(curve)
        for h in hist:
            hist_rows.append(dict(
                exp_id=EXP_ID, method=label, obs_ratio=ratio, schedule=sched,
                scenario_id=sid, cycle=h["cycle"], radius=h["radius_mean"],
                radius_raw=h.get("radius_raw", np.nan), J=h["J"],
                n_evals=h["n_evals"]))
        # Trajectory tier for every run: means and per-cycle metrics, with the
        # radius series appended so the estimate and what was actually used sit
        # in the same record.
        if traj is not None:
            metrics = dict(traj["metrics"])
            K = len(hist)
            if K:
                metrics["radius"] = np.array([h["radius_mean"] for h in hist])
                metrics["radius_raw"] = np.array(
                    [h.get("radius_raw", np.nan) for h in hist])
                metrics["J"] = np.array([h["J"] for h in hist])
                metrics["n_evals"] = np.array([h["n_evals"] for h in hist])
            snaps.add_trajectory(
                tag=f"traj/ratio{ratio}/{sched}/{label}/s{sid}",
                xb_mean=traj["xb_mean"], xa_mean=traj["xa_mean"],
                x_true=traj["x_true"], metrics=metrics,
                method=label, obs_ratio=ratio, schedule=sched,
                scenario_id=sid, n_cycles=int(traj["xa_mean"].shape[0]),
                admissible=admissible)

        if keep and snap is not None and snap["Xb"].size:
            hist_r = {h["cycle"]: h for h in hist}
            for j, step in enumerate(snap["steps"]):
                h = hist_r.get(int(step), {})
                op = scenarios[sid].operators[int(step)]
                snaps.add(
                    tag=f"ratio{ratio}/{sched}/{label}/step{int(step)}",
                    Xb=snap["Xb"][j], Xa=snap["Xa"][j],
                    x_true=scenarios[sid].truth_trajectory[int(step)],
                    obs_idx=np.asarray(op.indices, dtype=int),
                    y=np.asarray(scenarios[sid].observations[int(step)]),
                    r=np.full(scale.n_state, h.get("radius_mean", np.nan)),
                    r_raw=np.full(scale.n_state, h.get("radius_raw", np.nan)),
                    method=label, obs_ratio=ratio, schedule=sched,
                    scenario_id=sid, step=int(step),
                    time=float(snap["times"][j]),
                    radius=h.get("radius_mean", np.nan),
                    radius_raw=h.get("radius_raw", np.nan))

        flag = "  <-- FAILED" if out["failed"] else ""
        prog.step(f"p/n={ratio} {sched:<6s} {label:<34s} s={sid} "
                  f"rmse={out['rel_rmse_a']:.4f}{flag}")
    prog.done()

    df = pd.DataFrame(rows)
    ctx.save_table(df, "metrics.csv")
    if hist_rows:
        ctx.save_table(pd.DataFrame(hist_rows), "radius_history.csv")
    ctx.save_snapshots(snaps)
    if ctx.is_shard:
        ctx.finish(summary=dict(n_cells=len(df)))
        return

    # Every row is measured against the oracle of its own scenario. Comparing
    # against a regime average would let a lucky scenario look like a good
    # method, since the spread across scenarios is larger than the spread
    # across methods.
    ref = (df[df["method"].str.startswith("oracle")]
           .groupby(["obs_ratio", "schedule", "scenario_id"])["rel_rmse_a"]
           .mean().rename("rmse_oracle"))
    df = df.join(ref, on=["obs_ratio", "schedule", "scenario_id"])
    df["gap_pct"] = 100.0 * (df["rel_rmse_a"] / df["rmse_oracle"] - 1.0)

    summary = (df.groupby(["obs_ratio", "schedule", "method", "admissible"])
               .agg(rmse=("rel_rmse_a", "mean"), rmse_sd=("rel_rmse_a", "std"),
                    gap_pct=("gap_pct", "mean"),
                    radius=("radius_used", "mean"),
                    radius_raw=("radius_raw", "mean"),
                    drift=("drift_used", "mean"),
                    drift_raw=("drift_raw", "mean"),
                    evals=("evals_per_cycle", "mean"),
                    seconds=("elapsed_s", "mean"), n=("rel_rmse_a", "count"))
               .reset_index().sort_values(["obs_ratio", "schedule", "gap_pct"]))
    ctx.save_table(summary, "summary.csv")
    print()
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    hard = summary[summary["schedule"] == "random"]
    if hard.empty:
        hard = summary
    key_ratio = hard["obs_ratio"].min()
    hard = hard[hard["obs_ratio"] == key_ratio].sort_values("gap_pct")
    colors = ["lightgrey" if not a else "tab:blue" for a in hard["admissible"]]
    axes[0].barh(range(len(hard)), hard["gap_pct"], color=colors)
    axes[0].set_yticks(range(len(hard)))
    axes[0].set_yticklabels(hard["method"], fontsize=6)
    axes[0].set_xlabel("gap to the oracle of the same regime (%)")
    axes[0].set_title(f"(a) p/n = {key_ratio}, moving network\n"
                      "(grey: not admissible)")

    sub = df[df["method"].str.startswith("CV")]
    if not sub.empty:
        sc = axes[1].scatter(sub["drift_used"], sub["gap_pct"], s=22,
                             c=sub["obs_ratio"], cmap="viridis",
                             edgecolor="k", linewidth=0.3)
        fig.colorbar(sc, ax=axes[1], fraction=0.046, label="p/n")
        axes[1].set_xlabel("mean change of the radius between cycles")
        axes[1].set_ylabel("gap to the oracle (%)")
        axes[1].set_title("(b) the cost of a radius that moves")

    if hist_rows:
        hdf = pd.DataFrame(hist_rows)
        pick = hdf[(hdf["schedule"] == hard["schedule"].iloc[0]) &
                   (hdf["obs_ratio"] == key_ratio) &
                   (hdf["scenario_id"] == 0)]
        raw = pick[pick["method"] == "CV each cycle, raw"]
        if not raw.empty:
            axes[2].plot(raw["cycle"], raw["radius_raw"], lw=0.5, alpha=0.4,
                         color="k", label="raw estimate")
        for label in ["CV each cycle, raw", "CV each cycle, ewma 0.05",
                      "CV then freeze after 5"]:
            g = pick[pick["method"] == label]
            if not g.empty:
                axes[2].plot(g["cycle"], g["radius"], lw=1.2, label=label)
        axes[2].set_xlabel("assimilation cycle")
        axes[2].set_ylabel("radius in use")
        axes[2].set_title("(c) the estimate swings,\nthe parameter should not")
        axes[2].legend(fontsize=6)

    fig.suptitle(f"EXP-09: {n_cycles_run} cycles per run, "
                 f"{scale.n_scenarios} scenarios")
    fig.tight_layout()
    ctx.save_fig(fig, "cycles.png")

    ctx.save_latex(
        latex_table(summary[["obs_ratio", "schedule", "method", "rmse",
                             "gap_pct", "radius", "drift"]],
                    "Consecutive assimilation cycles. The gap is measured "
                    "against a fixed radius tuned by a sweep against the "
                    "truth, which an operational system could not compute.",
                    "tab:cycles"),
        "cycles.tex")

    adm = summary[summary["admissible"]]
    best = (adm.sort_values("gap_pct").iloc[0].to_dict() if not adm.empty
            else {})
    ctx.finish(summary=dict(n_cells=len(df),
                            n_failed=int(df["failed"].sum()),
                            oracle_radius={str(k): v for k, v in oracle_r.items()},
                            best_admissible=best))


if __name__ == "__main__":
    main(parse_cli())
