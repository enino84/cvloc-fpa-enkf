# -*- coding: utf-8 -*-
"""
Tests for the properties the results actually depend on.

These are not coverage tests. Each one guards a claim the paper makes, so that
a refactor that quietly breaks the claim fails the build rather than producing
a plausible wrong number. The Docker image runs them, so a broken image never
reaches a cluster.
"""
from __future__ import annotations

import numpy as np
import pytest

import cvloc
from cvloc import (CVObjective, CycleConfig, EnsembleSpace, Lorenz96CV,
                   gaspari_cohn, make_cycle, make_parameterization,
                   taper_weights)
from cvloc.cholesky import CholeskySpace
from cvloc.features import cluster, correlation_decay
from cvloc.metaheuristics import METAHEURISTICS, get_optimizer
from cvloc.taper import cyclic_distance, predecessors


@pytest.fixture(scope="module")
def space():
    return EnsembleSpace(make_cycle(CycleConfig(n=40, N=20), seed=7))


# ----------------------------------------------------------------------
# Taper
# ----------------------------------------------------------------------
def test_gaspari_cohn_known_values():
    assert gaspari_cohn(np.array([0.0]))[0] == pytest.approx(1.0)
    # The two branches must agree at z = 1, or the taper has a jump.
    assert gaspari_cohn(np.array([1.0]))[0] == pytest.approx(0.2083333, abs=1e-6)
    assert gaspari_cohn(np.array([2.0]))[0] == pytest.approx(0.0, abs=1e-12)
    assert gaspari_cohn(np.array([2.5, 10.0])).max() == 0.0


def test_gaspari_cohn_is_monotone_decreasing():
    z = np.linspace(0, 2, 400)
    y = gaspari_cohn(z)
    assert np.all(np.diff(y) <= 1e-12)


def test_cyclic_distance_wraps():
    d = cyclic_distance(40)
    assert d[0, 39] == 1.0
    assert d[0, 20] == 20.0
    assert np.allclose(d, d.T)


def test_predecessors_are_below_i_and_within_radius():
    n = 40
    for i in (1, 5, 20, 39):
        for r in (1, 3, 7):
            idx = predecessors(n, i, r)
            assert np.all(idx < i)
            assert np.all(cyclic_distance(n)[i, idx] <= r)


# ----------------------------------------------------------------------
# Ensemble space
# ----------------------------------------------------------------------
def test_radius_free_quantities_are_independent_of_r(space):
    """Y and d must not move when the radius changes. The whole cost argument
    of the paper rests on this."""
    Y0, d0 = space.Y.copy(), space.d.copy()
    for r in (0.5, 2.0, 12.0):
        space.analysis_mean(np.full(space.n, r))
    assert np.allclose(space.Y, Y0)
    assert np.allclose(space.d, d0)


def test_tiny_radius_reduces_to_pointwise_analysis(space):
    """With a radius far below one grid unit only the collocated observation
    is used, so each analysis point must be the scalar Kalman update."""
    xa = space.analysis_mean(np.full(space.n, 0.1))
    var = np.sum(space.X ** 2, axis=1)
    gain = var / (var + 1.0)
    expected = space.xb + gain * (space.cycle.y - space.xb)
    assert np.allclose(xa, expected, atol=1e-8)


def test_growing_radius_converges_to_the_unlocalized_filter(space):
    """Gaspari-Cohn equals one only at zero distance, so a large radius only
    approaches the unlocalized filter. ``r=None`` is the exact code path, and
    the approximation must converge to it."""
    exact = space.analysis_mean(None)
    errs = [np.max(np.abs(space.analysis_mean(np.full(space.n, r)) - exact))
            for r in (50.0, 500.0, 5000.0)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 1e-4


def test_analysis_ensemble_mean_matches_analysis_mean(space):
    r = np.full(space.n, 2.0)
    xa, Xa = space.analysis_ensemble(r)
    assert np.allclose(Xa.mean(axis=1), xa, atol=1e-9)
    assert np.allclose(space.analysis_mean(r), xa, atol=1e-9)


def test_analysis_reduces_error_against_the_background(space):
    bg = space.cycle.background_rmse()
    assert space.truth_error(np.full(space.n, 2.0)) < bg


def test_masked_observations_have_no_influence(space):
    """Withholding an observation must be exactly setting its weight to zero:
    an analysis with a mask must equal one computed without those rows."""
    mask = np.ones(space.p)
    mask[:5] = 0.0
    xa_masked = space.analysis_mean(np.full(space.n, 3.0), mask=mask)

    sub = cvloc.Cycle(Xb=space.cycle.Xb, x_true=space.cycle.x_true,
                      obs_idx=space.obs_idx[5:], y=space.cycle.y[5:],
                      r_inv=space.r_inv[5:], model=space.cycle.model)
    xa_sub = EnsembleSpace(sub).analysis_mean(np.full(space.n, 3.0))
    assert np.allclose(xa_masked, xa_sub, atol=1e-9)


# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------
def test_folds_are_frozen_across_candidates(space):
    """Two evaluations of the same radius must return exactly the same value.
    If the folds were redrawn per candidate, the search would be ranking
    different radii on different noise realizations."""
    obj = CVObjective(space, folds=10, rng=np.random.default_rng(0), budget=None)
    a = obj.raw(np.array([2.0]))
    obj.raw(np.array([7.0]))
    b = obj.raw(np.array([2.0]))
    assert a == b


def test_budget_is_enforced_and_cached(space):
    obj = CVObjective(space, folds=5, rng=np.random.default_rng(0), budget=4)
    for r in (1.0, 2.0, 3.0, 4.0):
        obj(np.array([r]))
    assert obj.n_evals == 4
    # A repeat is free and must not consume budget.
    obj(np.array([2.0]))
    assert obj.n_evals == 4 and obj.n_cache_hits == 1
    with pytest.raises(cvloc.BudgetExhausted):
        obj(np.array([9.0]))


def test_fork_shares_folds_but_resets_the_budget(space):
    obj = CVObjective(space, folds=5, rng=np.random.default_rng(0), budget=3)
    other = obj.fork(budget=10)
    assert other.n_evals == 0
    assert obj.raw(np.array([2.0])) == other.raw(np.array([2.0]))


def test_objective_never_touches_the_truth(space, monkeypatch):
    """The criterion must be computable without a reference trajectory. Poison
    the truth and check that J is unchanged."""
    obj = CVObjective(space, folds=10, rng=np.random.default_rng(0), budget=None)
    before = obj.raw(np.array([2.5]))
    space.cycle.x_true = np.full_like(space.cycle.x_true, np.nan)
    after = obj.raw(np.array([2.5]))
    assert np.isfinite(after) and after == before


# ----------------------------------------------------------------------
# Parameterizations
# ----------------------------------------------------------------------
def test_parameterizations_are_nested():
    """A uniform radius must be reachable by every K-block parameterization.
    EXP-07's whole diagnostic rests on this."""
    n = 40
    uni = make_parameterization("uniform", n)
    for K in (1, 2, 4, 8, 40):
        blocks = make_parameterization("blocks", n, K=K)
        assert np.allclose(blocks.expand(np.full(K, 3.0)),
                           uni.expand(np.array([3.0])))


def test_blocks_cover_every_grid_point_exactly_once():
    for K in (1, 3, 7, 40):
        b = make_parameterization("blocks", 40, K=K)
        assert b.assign.min() == 0 and b.assign.max() == K - 1
        assert np.bincount(b.assign).sum() == 40


def test_nested_parameterization_can_only_lower_the_attainable_minimum(space):
    """With K free radii the K=1 optimum is admissible, so the value of J at
    the uniform solution must be reproducible in the larger space."""
    obj1 = CVObjective(space, param=make_parameterization("uniform", space.n),
                       folds=5, rng=np.random.default_rng(0), budget=None)
    obj4 = CVObjective(space, param=make_parameterization("blocks", space.n, K=4),
                       folds=5, rng=np.random.default_rng(0), budget=None)
    assert obj1.raw(np.array([2.0])) == pytest.approx(
        obj4.raw(np.full(4, 2.0)), rel=1e-12)


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", METAHEURISTICS + ["random", "grid"])
def test_every_optimizer_respects_the_budget(space, name):
    budget = 25
    obj = CVObjective(space, folds=5, rng=np.random.default_rng(0),
                      budget=budget)
    theta, info = get_optimizer(name)(
        obj, np.array([0.5]), np.array([8.0]), budget,
        np.random.default_rng(1), **cvloc.defaults_for(name))
    assert obj.n_evals <= budget
    assert 0.5 <= float(theta[0]) <= 8.0
    assert np.isfinite(info["J"])


@pytest.mark.parametrize("name", METAHEURISTICS)
def test_every_optimizer_beats_a_single_random_draw(space, name):
    """A search that does not improve on its own starting point is broken."""
    budget = 40
    obj = CVObjective(space, folds=5, rng=np.random.default_rng(0),
                      budget=budget)
    theta, info = get_optimizer(name)(
        obj, np.array([0.5]), np.array([12.0]), budget,
        np.random.default_rng(2), **cvloc.defaults_for(name))
    assert info["J"] <= obj.trace[0] + 1e-12


def test_fpa_levy_scale_changes_the_step_size():
    """Yang's point: the reference Levy already carries a factor of 0.01, so
    dropping it multiplies the step by a hundred."""
    from cvloc.metaheuristics.base import levy
    rng = np.random.default_rng(0)
    small = np.abs(levy(1000, 1.5, rng, scale=0.01)).mean()
    rng = np.random.default_rng(0)
    big = np.abs(levy(1000, 1.5, rng, scale=1.0)).mean()
    assert big == pytest.approx(100.0 * small, rel=1e-9)


def test_search_is_deterministic_given_the_seed(space):
    out = []
    for _ in range(2):
        obj = CVObjective(space, folds=5, rng=np.random.default_rng(0),
                          budget=30)
        theta, info = get_optimizer("fpa")(
            obj, np.array([0.5]), np.array([8.0]), 30,
            np.random.default_rng(123), **cvloc.defaults_for("fpa"))
        out.append((float(theta[0]), info["J"]))
    assert out[0] == out[1]


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------
def test_correlation_decay_is_positive_and_bounded():
    Xb = make_cycle(CycleConfig(n=40, N=20), seed=3).Xb
    L = correlation_decay(Xb)
    assert L.shape == (40,)
    assert np.all(L > 0) and np.all(L <= 10)


def test_clustering_returns_one_label_per_grid_point():
    Xb = make_cycle(CycleConfig(n=40, N=20), seed=3).Xb
    labels, feats, info = cluster(Xb, K=None, K_range=(2, 5))
    assert labels.shape == (40,)
    assert feats.shape == (40, 2)
    assert 1 <= info["K"] <= 5
    param = make_parameterization("clusters", 40, labels=labels)
    assert param.expand(np.arange(param.K, dtype=float)).shape == (40,)


# ----------------------------------------------------------------------
# Modified Cholesky
# ----------------------------------------------------------------------
def test_cholesky_precision_is_symmetric_positive_definite():
    cyc = make_cycle(CycleConfig(n=40, N=20), seed=5)
    cs = CholeskySpace(cyc, r_max=8)
    for r in (1, 3, 6):
        B = cs.family.precision(np.full(40, float(r)))
        assert np.allclose(B, B.T, atol=1e-10)
        assert np.linalg.eigvalsh(B).min() > 0


def test_cholesky_landscape_is_piecewise_constant_in_the_radius():
    """The predecessor set changes only at integers, so 3.0 and 3.9 must give
    the identical estimator. That discreteness is the point of EXP-10."""
    cyc = make_cycle(CycleConfig(n=40, N=20), seed=5)
    cs = CholeskySpace(cyc, r_max=8)
    a = cs.analysis_mean(np.full(40, 3.0))
    b = cs.analysis_mean(np.full(40, 3.9))
    c = cs.analysis_mean(np.full(40, 4.0))
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def test_lorenz96_rk4_conserves_the_fixed_point():
    m = Lorenz96CV(n=40, F=8.0)
    x = np.full(40, 8.0)
    assert np.allclose(m.propagate(x, np.array([0.0, 1.0])), x, atol=1e-8)


def test_heterogeneous_forcing_is_per_component():
    F = cvloc.forcing_profile(40, "blocks", 5.0, 12.0, 4)
    m = Lorenz96CV(n=40, F=F)
    assert m.heterogeneous
    assert np.allclose(m.rhs(np.zeros(40)), F)


def test_climatological_ensemble_has_large_background_error():
    """If the ensemble is not climatological the sampling noise is small and
    localization stops mattering, which would quietly invalidate the regime the
    paper claims to study."""
    cyc = make_cycle(CycleConfig(n=40, N=20), seed=11)
    assert cyc.background_rmse() > 2.5


def test_analysis_registers_into_pyteda():
    from pyteda.analysis.registry import ANALYSIS_REGISTRY
    assert "letkf-cv" in ANALYSIS_REGISTRY


# ----------------------------------------------------------------------
# Smoothing and the bagged estimator
# ----------------------------------------------------------------------
def test_smoothing_modes_reduce_the_movement_of_the_radius():
    """The point of smoothing is a radius that moves less. If a mode does not
    reduce the cycle-to-cycle change, it is not doing its job."""
    from cvloc.analysis_cv_letkf import AnalysisLETKFCrossValidated as A

    class Dummy:
        def get_number_of_variables(self):
            return 40

    rng = np.random.default_rng(0)
    raw = np.exp(rng.normal(np.log(4.0), 0.9, size=60))   # noisy, centred on 4

    drifts = {}
    for mode, kw in (("none", {}), ("window", dict(window=5)),
                     ("ewma", dict(alpha=0.05)), ("freeze", dict(freeze_after=5))):
        a = A(Dummy(), smoothing=mode, **kw)
        used = np.array([float(a._smooth(np.array([v]))[0]) for v in raw])
        drifts[mode] = float(np.mean(np.abs(np.diff(used))))

    assert drifts["window"] < drifts["none"]
    assert drifts["ewma"] < drifts["window"]
    assert drifts["freeze"] < drifts["ewma"]


def test_freeze_holds_the_radius_after_its_horizon():
    from cvloc.analysis_cv_letkf import AnalysisLETKFCrossValidated as A

    class Dummy:
        def get_number_of_variables(self):
            return 40

    a = A(Dummy(), smoothing="freeze", freeze_after=4)
    out = [float(a._smooth(np.array([float(v)]))[0]) for v in (1, 2, 4, 8, 16, 32)]
    assert out[-1] == out[-2] == out[3]


def test_smoothing_is_geometric_not_arithmetic():
    """The radius is a scale, so estimates of 1 and 100 should average to 10,
    not to 50."""
    from cvloc.analysis_cv_letkf import AnalysisLETKFCrossValidated as A

    class Dummy:
        def get_number_of_variables(self):
            return 40

    a = A(Dummy(), smoothing="window", window=2)
    a._smooth(np.array([1.0]))
    assert float(a._smooth(np.array([100.0]))[0]) == pytest.approx(10.0)


def test_bagged_estimator_returns_one_vote_per_resample(space):
    from cvloc import select_radius_bagged
    theta, r, votes, info = select_radius_bagged(
        space, budget=12, n_resamples=6, opt_params=cvloc.defaults_for("fpa"))
    assert votes.shape == (6, 1)
    assert r.shape == (space.n,)
    assert info["n_resamples"] == 6
    lo, hi = cvloc.BOXES["wide"]
    assert lo <= float(theta[0]) <= hi


def test_bagged_resamples_change_the_split_not_the_network(space):
    """Only the partition moves between resamples. The observation network,
    and therefore Y and d, must be untouched."""
    from cvloc import select_radius_bagged
    Y0, d0, idx0 = space.Y.copy(), space.d.copy(), space.obs_idx.copy()
    select_radius_bagged(space, budget=10, n_resamples=4,
                         opt_params=cvloc.defaults_for("fpa"))
    assert np.allclose(space.Y, Y0)
    assert np.allclose(space.d, d0)
    assert np.array_equal(space.obs_idx, idx0)


def test_every_box_contains_the_optimum_of_the_partial_network_regime():
    """The informed box was calibrated for a fully observed network and did not
    contain the optimum once half the network was withheld, which turned a
    penalty against the criterion into a penalty against the box."""
    from cvloc import BOXES, CycleConfig, EnsembleSpace, make_cycle, truth_optimum
    for seed in (801, 802, 803):
        sp = EnsembleSpace(make_cycle(
            CycleConfig(n=40, N=20, network="uniform", dense_frac=0.5), seed))
        r_opt = truth_optimum(sp, 0.2, 20.0, 60)["r_opt"]
        lo, hi = BOXES["wide"]
        assert lo <= r_opt <= hi


# ----------------------------------------------------------------------
# Snapshot persistence
# ----------------------------------------------------------------------
def test_snapshot_round_trip(tmp_path, space):
    """What is written must come back identical. Everything a covariance panel
    needs has to survive the trip, or the archive is decoration."""
    from cvloc import SnapshotWriter, load_snapshots

    r = np.full(space.n, 2.5)
    xa, Xa = space.analysis_ensemble(r)
    w = SnapshotWriter().add(
        tag="t/0", Xb=space.cycle.Xb, Xa=Xa, x_true=space.cycle.x_true,
        obs_idx=space.obs_idx, y=space.cycle.y, r_inv=space.r_inv, r=r,
        cycle=0, rmse=space.rmse(xa))
    npz = tmp_path / "s.npz"
    w.write(str(npz), str(tmp_path / "s.csv"))

    recs, idx = load_snapshots(str(npz), str(tmp_path / "s.csv"))
    assert len(recs) == 1
    assert np.allclose(recs[0]["Xb"], space.cycle.Xb)
    assert np.allclose(recs[0]["Xa"], Xa)
    assert np.array_equal(recs[0]["obs_idx"], space.obs_idx)
    assert idx.loc[0, "cycle"] == 0


def test_snapshots_allow_a_ragged_observation_network(tmp_path):
    """With a moving network the number of observations differs between
    cycles, so the archive must not assume a common shape."""
    from cvloc import SnapshotWriter, load_snapshots

    w = SnapshotWriter()
    w.add(tag="a", obs_idx=np.arange(20), step=0)
    w.add(tag="b", obs_idx=np.arange(13), step=1)
    w.write(str(tmp_path / "s.npz"), str(tmp_path / "s.csv"))
    recs, _ = load_snapshots(str(tmp_path / "s.npz"))
    assert recs[0]["obs_idx"].size == 20
    assert recs[1]["obs_idx"].size == 13


def test_rebuild_covariances_matches_a_direct_computation(space):
    """The archive must reproduce the panels, not approximate them."""
    from cvloc import SnapshotWriter, rebuild_covariances
    from cvloc.taper import taper_matrix

    r = np.full(space.n, 3.0)
    rec = dict(Xb=space.cycle.Xb, r=r)
    out = rebuild_covariances(rec)
    assert np.allclose(out["B"], space.sample_covariance())
    assert np.allclose(out["C"], taper_matrix(space.n, r))
    assert np.allclose(out["BC"], out["B"] * out["C"])


def test_empty_writer_writes_nothing(tmp_path):
    from cvloc import SnapshotWriter
    assert SnapshotWriter().write(str(tmp_path / "s.npz")) is None


def test_trajectory_tier_round_trip(tmp_path):
    """The cheap tier must survive the trip with every metric intact: it is
    what every time series in the paper is drawn from."""
    from cvloc import SnapshotWriter, load_snapshots, per_cycle_frame

    K, n = 40, 12
    xa = np.random.default_rng(0).standard_normal((K, n))
    w = SnapshotWriter().add_trajectory(
        tag="run/a", xb_mean=xa + 0.1, xa_mean=xa, x_true=xa - 0.1,
        metrics=dict(rmse_a=np.arange(K, dtype=float),
                     radius=np.full(K, 3.0),
                     radius_raw=np.linspace(1.0, 9.0, K)),
        method="m", n_cycles=K)
    w.write(str(tmp_path / "t.npz"), str(tmp_path / "t.csv"))

    recs, idx = load_snapshots(str(tmp_path / "t.npz"), str(tmp_path / "t.csv"))
    assert idx.loc[0, "tier"] == "trajectory"
    frame = per_cycle_frame(recs[0])
    assert len(frame) == K
    assert set(["rmse_a", "radius", "radius_raw"]).issubset(frame.columns)
    assert np.allclose(frame["rmse_a"], np.arange(K))
    # float32 is deliberate: these are for plotting, not for restarting a filter
    assert recs[0]["xa_mean"].dtype == np.float32


def test_the_two_tiers_are_distinguishable_in_the_index(tmp_path):
    from cvloc import SnapshotWriter, load_snapshots
    w = SnapshotWriter()
    w.add(tag="e", Xb=np.zeros((5, 3)))
    w.add_trajectory(tag="t", xa_mean=np.zeros((4, 5)),
                     metrics=dict(rmse_a=np.zeros(4)))
    w.write(str(tmp_path / "s.npz"), str(tmp_path / "s.csv"))
    _, idx = load_snapshots(str(tmp_path / "s.npz"), str(tmp_path / "s.csv"))
    assert sorted(idx["tier"]) == ["ensemble", "trajectory"]


def test_analysis_records_state_means_per_cycle():
    """The means come from the analysis object, since pyteda keeps per-cycle
    states only in its legacy mode."""
    from cvloc.analysis_cv_letkf import AnalysisLETKFCrossValidated as A

    class Dummy:
        def get_number_of_variables(self):
            return 40

    a = A(Dummy(), store_states=True)
    assert a.state_means[0].shape == (0, 40)
    a.xb_history.append(np.zeros(40, dtype=np.float32))
    a.xa_history.append(np.ones(40, dtype=np.float32))
    xb, xa = a.state_means
    assert xb.shape == (1, 40) and xa.shape == (1, 40)
