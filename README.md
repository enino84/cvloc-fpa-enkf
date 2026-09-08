# Cross-validated estimation of the localization radius

Companion code for *Cross-Validated Estimation of the Localization Radius in
Ensemble Data Assimilation* (Nino-Ruiz and Yang).

Covariance localization is required in ensemble data assimilation, and its
radius of influence is normally chosen by tuning against a known reference
trajectory, which an operational system does not have. This suite treats the
radius as the solution of an optimization problem whose objective uses
**observations only**: the observation network is split into training and
validation subsets, the analysis is computed from the training subset alone,
and the resulting state is scored against the withheld observations. No
reference trajectory enters the objective.

The objective is derivative free, evaluated under sampling noise, and cheap: in
ensemble space the projected anomalies `Y = HX` and the innovation `d` do not
depend on the radius, so a candidate is scored without any model integration.
It is minimized with the Flower Pollination Algorithm, against a field of
competitors and controls.

---

## What is in here

```
cvloc/                     the method
  taper.py                 Gaspari-Cohn, ring geometry, predecessor sets
  model.py                 Lorenz 96, RK4, heterogeneous forcing and networks
  ensemble_space.py        R-localized analysis; the cycle object
  parameterization.py      how many free radii and how they map to the grid
  objective.py             the cross-validated criterion, and the oracle
  features.py              ensemble features and the clustering built on them
  cholesky.py              the radius as a predecessor set of a regression
  cycles.py                cycle generation and the select-then-analyze driver
  analysis_cv_letkf.py     pyteda analysis class, registered as `letkf-cv`
  metaheuristics/          FPA, SA, PSO, DE, GA, firefly + random/grid/local
experiments/               one script per experiment, stable identifiers
scripts/                   runner, table assembly, shard merge
tests/                     54 tests, each guarding a claim the paper makes
```

`pyteda` comes from PyPI, pinned to an exact version in `requirements.txt`; the
version used is recorded in every `manifest.json`.

## Quick start

```bash
make build                 # builds the image and runs the tests
make smoke                 # whole suite in a few minutes
make findings SCALE=smoke  # read the digest
```

Without Docker:

```bash
pip install -r requirements.txt
make local-test
make local-all SCALE=smoke
```

Three scales. `smoke` validates the pipeline in minutes, `quick` is for
iterating on a change, `paper` is the configuration the article reports.
Everything honours `SCALE`, so `make exp07 SCALE=quick` runs one experiment at
one scale.

## The experiments

Each has a stable identifier that names its output directory, appears in every
CSV row, and is cited by the generated LaTeX, so any number in the paper can be
traced to the run that produced it.

| id | question | feeds |
|---|---|---|
| `EXP-00-TUNING` | Are the search hyperparameters calibrated fairly, against `J` alone, on held-out cycles? | appendix; writes `results/frozen_params.json`, which every other experiment reads |
| `EXP-01-LANDSCAPE` | Does the radius matter, and do the truth-based and cross-validated curves agree? | the spread, the asymmetry, the covariance panels |
| `EXP-02-BASELINES` | Background, no localization, fixed radius, CV, oracle, per-variable | the main table |
| `EXP-03-CVPROXY` | Is the criterion a good stand-in for the truth? What does it cost, and is it biased? | the headline result |
| `EXP-04-META` | How do the searches compare at equal budget on the tapered problem? | the comparison table |
| `EXP-05-BUDGET` | How much search does the criterion need? Does a longer run or a larger population help? | convergence figure |
| `EXP-06-ENSEMBLE` | Does the criterion reproduce the growth of the optimal radius with `N`? | independent check |
| `EXP-07-KRADII` | Is the degradation with more free radii a property of the parameterization or a failure of the search? | the negative result, taken apart |
| `EXP-08-HETERO` | Where grid points are *not* exchangeable, does a spatially varying radius pay? | the positive counterpart |
| `EXP-09-CYCLES` | Over consecutive cycles with a moving network, how must the per-cycle estimate be combined? | **the headline result** |
| `EXP-10-CHOLESKY` | On a discrete, multimodal landscape, does the choice of search matter? | the comparison that discriminates |
| `EXP-11-SMOOTHER` | A criterion that every radius can reach: augmented state `(k-1, k)`, scored on observations the analysis never assimilated | **the criterion that works where holding out does not** |

## The result the suite is built around

The criterion identifies the right radius. The raw per-cycle estimate still
loses.

In a run with half the network observed and relocated at random every cycle, a
sweep of fixed radii put the cycled optimum at eight; the criterion selected
7.96 on average, essentially exact; and it produced an analysis 38% worse than
simply holding the radius at eight. The diagnostic is the mean absolute change
of the radius between consecutive cycles, which was 7.67. The estimate averaged
eight while never being eight, swinging across the box from one cycle to the
next. The landscape is flat near its minimum, so the arg-min of one noisy
realization moves a great deal even when the quantity being estimated does not,
and the analysis pays for every swing.

The radius is therefore not a parameter to be estimated afresh each cycle. It
is a slowly varying quantity of which each cycle provides a noisy estimate.
Treated that way, with the estimates combined over time, the criterion reaches
the oracle tuning without any access to the truth. `EXP-09` varies how they are
combined (`none`, `window`, `ewma`, `freeze`) against a fixed radius a
practitioner might guess and a fixed radius tuned by a sweep against the truth.

A caution when reading small-scale runs: the smoothing horizon must be short
relative to the run. At `smoke` the runs are 20 cycles and an `ewma` with
alpha = 0.05 needs about that many just to reach its target, so the ordering
among smoothing variants there is meaningless. Use `quick` (100 cycles) or
`paper` (300) to compare them.

## Why holding observations out is not enough

The criterion used by `EXP-01` through `EXP-09` withholds part of the network
and scores the analysis against what it withheld. That construction has a
defect which is structural rather than a matter of tuning: the analysis at a
held-out location depends only on the radius **at that location**, so when the
network is sparse the radii of unobserved points do not enter the objective at
all.

Measured directly in this suite, with half of forty points observed: moving the
twenty radii of unobserved points from 3 to 15 left the objective unchanged to
eight decimal places, while the true error went from 1.97 to 3.65. Half the
parameters were invisible to the criterion. That alone explains why every
spatially varying parameterization loses in `EXP-02`, `EXP-07` and `EXP-08`,
without needing an argument about search or about exchangeability.

Two other routes were measured and rejected. Scoring against the full network
is circular: a short radius makes each point fit its own observation and the
objective falls monotonically to the left edge of the box. Scoring a forecast
against the next cycle's observations would work, but at cycle `k` those
observations do not exist yet.

`EXP-11` takes a different route. The state is the pair `(k-1, k)`; the
modified Cholesky factorization is estimated on the joint vector, so its
cross-block couples every component of `k-1` to the components of `k`; only the
observations of `k-1` are assimilated; and the radius is scored on the `k`
block against the observations of `k`, which that analysis never saw. Nothing
is withheld, the whole network is used, no future data is needed, and every
radius reaches the block where the score is taken. Switching the temporal
predecessors off flattens the criterion completely, which is the direct
evidence that the cross-block is what carries the signal.

## The regimes

Four axes control whether the radius is a real decision, and the suite varies
all four.

**Observation density** `p/n` in {1, 1/2, 1/4}. This is the axis the draft
never varied, and it turns out to be decisive. With a fully observed network
the analysis is essentially "take the observation": information never has to
travel, so the radius only controls how much sampling noise is let in, the
optimum sits near the left edge of the box, and the problem is one-sided. With
half the network the gain from letting information travel rises from 1-3% to
10-12% while the gain from filtering stays near 50%, the optimum moves into the
interior, and the radius becomes a genuine two-sided trade-off.

**Observation schedule**, `fixed` against `random`. A real network moves:
satellite swaths, station dropouts, cloud cover. With `random` the observed
locations are redrawn every cycle, which is both more realistic and, it turns
out, easier for the filter, since no grid point stays unobserved forever. It is
also the honest setting for the question the paper asks, because when the
network moves there is no constant that is right for every cycle.

**Ensemble size** `N` in {5, 10, 20, 40}, extended downward: small ensembles
with a partial network are where localization is decisive.

**Heterogeneity**, in `EXP-08`: forcing and observation density varying across
the domain, which is the only setting in which a spatially varying radius can
pay.

## Two estimators

`select_radius` implements the criterion as the draft writes it: average `J`
over the folds, minimize the average, one arg-min.

`select_radius_bagged` redraws the train/validation split `B` times, minimizes
each realization separately, and reports the *mode* of the arg-minima. On a
flat landscape a mode only needs each vote in the right bin rather than the
exact minimum, so it degrades more gracefully, and it returns a distribution of
radii instead of a point. The observation network does not change between
resamples, only the partition of it, so `B` resamples cost `B` searches and not
one model integration. Carrying the population from one resample to the next
recovers most of the benefit at a third of the evaluations, which is the one
argument for a population that does not depend on the shape of the landscape.

## Recreating the figures without rerunning anything

Two tiers are archived, because the two questions have very different costs.

The **trajectory tier** is kept for every run: the analysis and background
means at every cycle, the truth, and twelve per-cycle metrics (absolute and
relative RMSE for both, ensemble spread, CRPS, the radius in use and the raw
estimate before smoothing, the attained objective, the evaluation count). Means
are `float32`, since they are for plotting rather than for restarting a filter.
That is about 140 kB for a 300-cycle run, cheap enough to keep for everything,
and it is what every time series in the paper is drawn from. It also gives
spread against error per cycle, which no metrics table carried before.

The **ensemble tier** is kept for a few cycles of a few methods: the full
forecast and analysis ensembles, which is the only thing a covariance panel can
be rebuilt from, and two orders of magnitude larger per cycle.

A metrics table answers how well a method did. It cannot answer what the
estimator looked like, which is what every covariance panel, taper plot and
radius profile is actually asking. Those need the ensemble, and an ensemble
does not fit in a row.

`EXP-01`, `EXP-08` and `EXP-09` therefore write `snapshots.npz` alongside their
CSVs, with `snapshots_index.csv` recording what each slot holds. Each record
carries the forecast and analysis ensembles, the truth, the observed indices,
the observations and their precisions, and both the radius in use and the raw
estimate before smoothing.

The observed indices matter more than they look. With `schedule='random'` the
network moves every cycle, so neither the ensemble nor the network can be
reconstructed from a seed without replaying the whole scenario. Storing them is
the difference between a figure that can be redrawn and one that cannot.

```python
from cvloc import load_snapshots, per_cycle_frame, rebuild_covariances

recs, idx = load_snapshots("results/EXP-09-CYCLES_paper/snapshots.npz",
                           "results/EXP-09-CYCLES_paper/snapshots_index.csv")

# a time series: any run, every cycle
row    = idx.query("tier == 'trajectory' and method == 'CV each cycle, raw'").iloc[0]
series = per_cycle_frame(recs[int(row.slot)])     # rmse, spread, crps, radius...

# a covariance panel: one cycle, full ensembles
row = idx.query("tier == 'ensemble' and step == 50").iloc[0]
cov = rebuild_covariances(recs[int(row.slot)])    # B, the taper C, and B * C
```

`rebuild_covariances` performs exactly what the panels of Figure 1 perform, so
a panel can be restyled, or recomputed with a different taper, from the archive
alone. Multi-cycle runs store a few cycles per run rather than all of them
(`Scale.store_states_at` gives the fractions), and only for the methods a
figure would compare, which keeps the archive at megabytes instead of hundreds.

## Three things the code is built to make honest

**The criterion never sees the truth.** `CVObjective` is computed from the
ensemble, the observations and the folds. A test poisons `x_true` with `NaN`
and asserts that `J` is unchanged. The truth-based optimum exists in the code
as `TruthObjective` and is labelled an oracle everywhere it appears; it is
never used to select anything a method reports.

**Equal budget means equal budget.** Every optimizer receives the same
objective object, over the same frozen folds, from the same ensemble. Unique
evaluations are counted, repeats are cached and counted separately, and a hard
call limit stops an optimizer that keeps reproposing a point it has already
seen. `fork()` is what makes a head-to-head reproducible.

**The oracle is swept per scenario.** Tuning a fixed radius on one trajectory
and applying it to another does not give an upper bound: an admissible rule can
then beat it and a meaningless negative gap appears. Each scenario in `EXP-09`
gets the radius that is actually best for it.

**The admissible box contains the optimum of every regime.** The box was
originally calibrated for a fully observed network and did not contain the
optimum once half the network was withheld, which turned a penalty against the
criterion into a penalty against the box. A test now asserts the containment.

**The folds are frozen per cycle.** Resampling per candidate would rank
different radii on different noise realizations, and the ranking is what the
search consumes. A test asserts that evaluating the same radius twice returns
exactly the same number.

## A note on the parameterization sweep

The parameterizations are **nested**: a uniform radius is a `K`-block radius
with all blocks equal, so the attainable minimum of `J` is non-increasing in
`K` by construction. That gives a sharp test, and `EXP-07` is built around it:

- attained `J` **rises** with `K` &rarr; the search failed, since the `K = 1`
  solution was admissible at every larger `K` and was not found;
- attained `J` **falls** while the true error rises &rarr; the criterion is
  being overfitted;
- neither moves &rarr; the freedom is neither exploited nor harmful in `J`.

`EXP-07` crosses four factors so that each remedy can be switched on and off
independently: the number of free radii, a fixed against a `K`-scaled budget,
a wide against an informed admissible box, and the broadcast move that shares
the incumbent's radius across components. The `verdict` column applies the test
above to every cell.

## A note on the Flower Pollination implementation

Three details in `cvloc/metaheuristics/fpa.py` differ from a naive reading of
the equations, and each is documented in the module:

- **Step size.** In the reference implementation `Levy(d)` already carries a
  factor of `0.01`, so the effective global step is `gamma * 0.01 * step`. An
  implementation that drops it and uses `gamma = 1` takes steps a hundred times
  larger than intended. `gamma` and `levy_scale` are separate parameters here
  so the two can be varied independently; `EXP-00` sweeps `gamma`.
- **Switching probability.** The reference performs the *global* Levy step when
  `rand > p`, so `p = 0.8` means global pollination one time in five, not four.
  `switch_is_global` makes the convention explicit.
- **Direction.** The reference computes `x_i - x_best`; the draft writes
  `r_best - r`. Mantegna's step is symmetric so the two agree in distribution.
  `toward_best` selects the other form.

## Reproducibility

Every run writes a `manifest.json` with the full configuration, the seeds, the
package versions, the platform and the wall time. The Docker image pins one
BLAS thread and `PYTHONHASHSEED=0`, without which runs are reproducible only up
to thread scheduling. Two runs with the same manifest must produce the same
numbers; if they do not, that is a bug.

## Leaving the definitive runs going overnight

```bash
tmux new -s cvloc                  # or: make overnight SCALE=paper
bash scripts/overnight.sh paper
```

`overnight.sh` runs the unit tests first, then the whole suite at `smoke`
scale, and only then starts `paper`. A configuration error therefore surfaces
in five minutes instead of at hour six. Set `SKIP_PRECHECK=1` to skip that.
Expect about 7.5 hours single-threaded and roughly 50 MB of snapshot archives.
The log is `results/overnight_paper_<timestamp>.log` and the digest is printed
at the end.

## Running on a cluster

```bash
make shards N=8 SCALE=paper     # calibrates once, then eight containers
make shard-status
make merge N=8 SCALE=paper      # concatenates and rebuilds the tables
```

Shards partition the cell list round-robin, so the split is deterministic and
independent of execution order. Figures are not rebuilt from shards: a figure
drawn from a partial slice is worse than no figure, so rerun unsharded when a
figure is needed.

`METHODS` filters by substring across all experiments, which is the fast way to
rerun one optimizer after a change:

```bash
make exp04 SCALE=quick METHODS=fpa,random
```
