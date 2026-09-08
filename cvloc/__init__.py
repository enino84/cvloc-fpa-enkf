# -*- coding: utf-8 -*-
"""
cvloc: cross-validated estimation of the localization radius in ensemble data
assimilation, solved with the Flower Pollination Algorithm.

Importing this package registers the analysis method ``letkf-cv`` into pyteda's
global registry. Nothing in pyteda is patched or shadowed.

Caveat on parallelism: joblib workers spawned by ``pyteda.experiments.Benchmark``
do not import this package, so the registry entry is missing there. Run the
Benchmark with ``parallel=False`` and parallelize across experiments or shards
instead, which is what ``scripts/run_all.sh`` does.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import features, persist, progress, taper
from .analysis_cv_letkf import AnalysisLETKFCrossValidated
from .cholesky import CholeskyPrecision, CholeskySpace
from .cycles import (CycleConfig, cv_sweep, make_cycle, make_cycles,
                     select_radius, select_radius_bagged, truth_optimum)
from .ensemble_space import Cycle, EnsembleSpace
from .metaheuristics import (CONTROLS, DEFAULTS, LABELS, METAHEURISTICS,
                             OPTIMIZERS, defaults_for, get_optimizer)
from .model import Lorenz96CV, forcing_profile, observation_network
from .objective import BudgetExhausted, CVObjective, TruthObjective, make_folds
from .persist import (SnapshotWriter, load_snapshots, per_cycle_frame,
                      rebuild_covariances)
from .parameterization import BOXES, Blocks, Clusters, PerVariable, Uniform, box
from .parameterization import make as make_parameterization
from .taper import (cyclic_distance, gaspari_cohn, obs_distance, predecessors,
                    taper_matrix, taper_weights)

__all__ = [
    "AnalysisLETKFCrossValidated",
    "BOXES", "Blocks", "BudgetExhausted",
    "CONTROLS", "CVObjective", "CholeskyPrecision", "CholeskySpace",
    "Clusters", "Cycle", "CycleConfig",
    "DEFAULTS", "EnsembleSpace", "LABELS", "Lorenz96CV",
    "METAHEURISTICS", "OPTIMIZERS", "PerVariable", "TruthObjective", "Uniform",
    "box", "cv_sweep", "cyclic_distance", "defaults_for", "features",
    "forcing_profile", "gaspari_cohn", "get_optimizer", "make_cycle",
    "make_cycles", "make_folds", "make_parameterization", "obs_distance",
    "observation_network", "predecessors", "progress", "select_radius",
    "SnapshotWriter", "load_snapshots", "per_cycle_frame", "persist",
    "rebuild_covariances",
    "select_radius_bagged", "taper", "taper_matrix", "taper_weights", "truth_optimum",
    "__version__",
]
