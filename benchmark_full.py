"""
TRA Algorithm Full Benchmark Suite (research-paper grade)
==========================================================

Purpose
-------
Benchmark TRA (tra_algorithm.core.EnhancedTRA, exported as OptimizedTRA) against
the most commonly used ML algorithms in the literature, across a broad,
publication-appropriate dataset suite, with the full metric set and statistical
rigor a research paper's experiments section needs.

WHY THE TRA CONFIGURATION CHANGED FROM THE PREVIOUS VERSION OF THIS FILE
--------------------------------------------------------------------------
The previous version of this script instantiated TRA with essentially every
experimental flag turned on simultaneously (feature_selection=True,
calibrate_output=True, cluster_experts=True, intelligent_routing=True,
router_loss_aware=True, diversity_reweighting=True, calibrate_tracks=True,
top_k=3, ...). That is the OPPOSITE of "use TRA efficiently": several of these
are not just unvalidated, they are DOCUMENTED to hurt TRA on this project's own
prior benchmarking --

  - feature_selection=True enables a univariate SelectKBest filter that is
    structurally hostile to a mixture-of-experts (a feature can be predictive
    within each regime while its GLOBAL correlation is near zero) -- measured
    cost: 22 accuracy points on a regime-structured classification set, 67%
    higher MAE on a piecewise regression set. core.py's own default is now
    False for exactly this reason.
  - calibrate_output=True measured as TRA's WORST classification configuration
    (0.7584 vs 0.7764 accuracy) without even improving ECE, because it spends
    training data on a calibrator TRA's fused probabilities don't need.
  - intelligent_routing / cluster_experts / router_loss_aware / top_k>1 are
    part of an experimental hierarchical-routing subsystem that has never been
    benchmarked against the validated routing/stacking configuration below,
    and top_k specifically has NO EFFECT AT ALL unless intelligent_routing is
    also on (the default soft-routing path blends over every track already).
  - The old `expert_capacity=1.5` value overrides TRA's own auto-computed
    default (n_samples / n_tracks) with a number that has no relationship to
    the dataset. It happens to be inert (nothing in core.py currently gates on
    Track.capacity_violations), but it is not "efficient use" -- it is a
    leftover placeholder.

This version instead uses the SAME evidence-based TRA configuration that
produced this project's actual, published-quality benchmark numbers (see
benchmark.py's docstring for the full derivation), which core.py's own
defaults now also reflect after a subsequent round of fixes:
  - n_tracks=4 (RandomForest + LightGBM + XGBoost + SVM: heterogeneous experts
    that are also the strongest available baselines, so TRA is compared
    against models it structurally contains).
  - combination_mode is tested as THREE separate variants -- "TRA (stacking)",
    "TRA (routing)" (with router_use_oof=True), and "TRA (auto)" (lets TRA's
    own internal CV probe pick routing / stacking / flat_average per dataset,
    which is the most direct way to test whether TRA's regression weak spot
    on regime-free data is fixable without any code change) -- entered as
    separate model rows so the comparison shows which fusion actually earns
    its keep on each dataset, rather than asserting one is universally best.
  - feature_selection=False, small_data_auto_mode=False (required so the
    stacking/routing variants are not silently collapsed into the same model
    on any dataset under 2000 rows), calibrate_output=False.
  - Everything else is left at core.py's own defaults, which already include
    n_estimators=100 (matches the RandomForest/XGBoost/LightGBM baselines
    below instead of training with half their capacity), meta_learner_cv=True
    with log-loss-scored regularization selection, and stacking_use_signals=True
    (the structural signal extractor's 5 signals are fed into the stacking
    meta-learner, not just the router) -- three fixes validated on a full
    20-dataset before/after run to improve accuracy, ROC-AUC, log loss and ECE
    with no regressions. Re-deriving or re-overriding these here would silently
    undo that validation.

DATASETS (20: the same research-paper-appropriate suite used in benchmark.py)
------------------------------------------------------------------------------
Classification (10): Breast Cancer Wisconsin, Wine, Digits, Ionosphere, Sonar,
Pima Indians Diabetes, Vehicle Silhouettes, Banknote Authentication, Spambase,
Glass Identification. (Iris is deliberately excluded -- every model scores
~97-100% on it, so it has no power to discriminate a good algorithm from a
great one.)
Regression (10): Diabetes, California Housing, Friedman #1/#2/#3 (the classic
synthetic stress-tests from the ensemble-learning literature), Wine Quality
(Red), Concrete Compressive Strength, Auto MPG, Yacht Hydrodynamics, and a
synthetic high-dimensional linear-regression set.
Real (non-toy) sets are pulled once via sklearn's built-in loaders or a pinned
`fetch_openml` call and cached under ~/scikit_learn_data afterwards; a set that
fails to download on a fresh machine is skipped with a message, not fatal.

MODELS (17, or 18 with --include-difficulty-aware)
----------------------------------------------------
3 TRA variants (above) plus 14 baselines spanning linear models
(Logistic/Ridge), instance-based (KNN), single/bagged trees (RandomForest,
ExtraTrees), three DISTINCT boosting families (GradientBoosting,
HistGradientBoosting, and -- if installed -- XGBoost/LightGBM/CatBoost), a
kernel method (SVM/SVR), a neural net (MLP), classic boosting (AdaBoost), and
two hand-built meta-ensembles (Voting, Stacking) over the strongest available
tree/boosting learners -- i.e. exactly the roster a reviewer expects TRA to be
measured against, including a "fair fight" against an ensemble built from the
same model families TRA's own tracks use.

METRICS
-------
Classification: accuracy, weighted/macro F1, ROC-AUC (binary or one-vs-rest),
log loss, multiclass Brier score, Expected Calibration Error (ECE, equal-mass
binning). The last three exist specifically because TRA's validated advantage
over baselines is calibration, not raw accuracy -- a benchmark that only
reports accuracy/F1 cannot see that finding at all.
Regression: RMSE, MAE, R^2.
Every model sees IDENTICAL cross-validation folds per dataset (one `cv.split`
call, fixed random_state, model cloned fresh per fold), so per-fold scores are
PAIRED -- this script therefore also runs a paired t-test of every TRA variant
against the strongest non-TRA baseline on each dataset (see
`compute_paired_significance`), plus an overall BETTER/WORSE/ns tally across
all datasets, and a Demsar-style average-rank table across the whole suite.

USAGE
-----
    python benchmark_full.py                        # full 20-dataset, 5-fold run
    python benchmark_full.py --quick                 # fast sanity check (<=300 rows/dataset, 3-fold)
    python benchmark_full.py --datasets Wine Digits "Auto MPG"
    python benchmark_full.py --folds 5 --max-samples 5000
    python benchmark_full.py --include-difficulty-aware   # adds a 4th TRA variant
    python benchmark_full.py --skip-online            # skip the partial_fit/drift experiment
    python benchmark_full.py --skip-plots
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sklearn.base import clone
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_regression,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = Path("benchmark_outputs")


# ---------------------------------------------------------------------------
# Optional external boosting libraries
# ---------------------------------------------------------------------------

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


# ---------------------------------------------------------------------------
# TRA import
# ---------------------------------------------------------------------------

try:
    from tra_algorithm.core import OptimizedTRA
    TRA_IMPORT_ERROR = None
except Exception as exc:
    OptimizedTRA = None
    TRA_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Dataset definitions (the same 20-dataset research suite used in benchmark.py)
# ---------------------------------------------------------------------------

def _openml(name: str, version="active", target: str = "label"):
    """Build a zero-arg loader for a pinned OpenML dataset.

    target="label" -> string/categorical target, LabelEncoder'd to 0..K-1 (classification).
    target="float" -> numeric target OpenML may hand back as an object/string dtype; cast
                       straight to float (regression).
    """
    def _load():
        d = fetch_openml(name, version=version, as_frame=False, parser="auto")
        X = np.asarray(d.data, dtype=float)
        y = np.asarray(d.target)
        y = LabelEncoder().fit_transform(y) if target == "label" else y.astype(float)
        return X, y
    return _load


def _load_auto_mpg():
    # 6 of 398 rows have a missing 'horsepower' value; dropped (<2% of the data, and this is a
    # fixed evaluation set, not a production pipeline) rather than imputed.
    d = fetch_openml("autoMpg", as_frame=False, parser="auto")
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=float)
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


CLASSIFICATION_DATASETS: List[Tuple[str, Any]] = [
    ("Breast Cancer", load_breast_cancer),
    ("Wine", load_wine),
    ("Digits", load_digits),
    ("Ionosphere", _openml("ionosphere")),
    ("Sonar", _openml("sonar")),
    ("Pima Diabetes", _openml("diabetes", version=1)),
    ("Vehicle", _openml("vehicle")),
    ("Banknote Auth", _openml("banknote-authentication")),
    ("Spambase", _openml("spambase")),
    ("Glass", _openml("glass")),
]

REGRESSION_DATASETS: List[Tuple[str, Any]] = [
    ("Diabetes", load_diabetes),
    ("California Housing", fetch_california_housing),
    ("Friedman1", lambda: make_friedman1(n_samples=1000, n_features=10, noise=1.0,
                                        random_state=RANDOM_STATE)),
    ("Friedman2", lambda: make_friedman2(n_samples=1000, noise=1.0, random_state=RANDOM_STATE)),
    ("Friedman3", lambda: make_friedman3(n_samples=1000, noise=1.0, random_state=RANDOM_STATE)),
    ("Wine Quality Red", _openml("wine-quality-red", target="float")),
    ("Concrete Strength", _openml("Concrete_Compressive_Strength", target="float")),
    ("Auto MPG", _load_auto_mpg),
    ("Yacht Hydrodynamics", _openml("yacht_hydrodynamics", target="float")),
    ("Synthetic Regression", lambda: make_regression(n_samples=1000, n_features=20,
                                                     n_informative=15, noise=10.0,
                                                     random_state=RANDOM_STATE)),
]

DATASET_LOADERS: Dict[str, Tuple[str, Any]] = {}
for _name, _loader in CLASSIFICATION_DATASETS:
    DATASET_LOADERS[_name] = ("classification", _loader)
for _name, _loader in REGRESSION_DATASETS:
    DATASET_LOADERS[_name] = ("regression", _loader)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def log(msg: str = "") -> None:
    print(msg, flush=True)


def safe_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return None


def load_dataset(name: str, max_samples: int) -> Tuple[np.ndarray, np.ndarray, str]:
    task, loader = DATASET_LOADERS[name]
    data = loader()

    if isinstance(data, tuple):
        X, y = data
    else:
        X, y = data.data, data.target

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if max_samples > 0 and len(X) > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)

        if task == "classification":
            # Stratified subsampling (per-class capped, then topped up) so rare classes (e.g.
            # Glass) survive even under an aggressive --quick cap.
            selected = []
            classes = np.unique(y)
            per_class = max(1, max_samples // len(classes))

            for cls in classes:
                cls_idx = np.flatnonzero(y == cls)
                take = min(per_class, len(cls_idx))
                selected.extend(
                    rng.choice(cls_idx, size=take, replace=False).tolist()
                )

            selected = np.asarray(selected, dtype=int)
            if len(selected) < max_samples:
                remaining = np.setdiff1d(
                    np.arange(len(X)), selected, assume_unique=False
                )
                extra = rng.choice(
                    remaining,
                    size=min(max_samples - len(selected), len(remaining)),
                    replace=False,
                )
                selected = np.concatenate([selected, extra])

            selected = selected[:max_samples]
        else:
            selected = rng.choice(len(X), size=max_samples, replace=False)

        X = X[selected]
        y = y[selected]

    return X, y, task


# ---------------------------------------------------------------------------
# TRA configuration -- validated, evidence-based settings only (see module
# docstring for why every experimental flag from the previous version of this
# file was removed rather than "more is more"-tuned).
# ---------------------------------------------------------------------------

TRA_VARIANT_DISPLAY_NAMES = {
    "stacking": "TRA (stacking)",
    "routing": "TRA (routing)",
    "auto": "TRA (auto)",
    "routing-difficulty-aware": "TRA (routing, difficulty-aware)",
}


def get_tra_kwargs(task_type: str, variant: str) -> Dict[str, Any]:
    """Return the constructor kwargs for one named, evidence-based TRA variant.

    Only combination_mode (and, for "routing", router_use_oof / difficulty_aware_fusion) differs
    between variants; every other setting is either a validated override (feature_selection=False,
    small_data_auto_mode=False, calibrate_output=False -- all documented in this module's
    docstring) or is deliberately left at core.py's own default rather than re-specified here, so
    a future core.py improvement is inherited automatically instead of being silently shadowed by
    a stale copy of the value in this script.
    """
    common: Dict[str, Any] = {
        "task_type": task_type,
        "n_tracks": 4,
        "random_state": RANDOM_STATE,
        "small_data_auto_mode": False,
        "feature_selection": False,
        "calibrate_output": False,
        "n_jobs": -1,
    }

    if variant == "stacking":
        return {**common, "combination_mode": "stacking"}
    if variant == "routing":
        return {**common, "combination_mode": "routing", "router_use_oof": True}
    if variant == "routing-difficulty-aware":
        return {**common, "combination_mode": "routing", "router_use_oof": True,
                "difficulty_aware_fusion": True}
    if variant == "auto":
        # Lets TRA's own internal CV probe (see core.py's _select_combination_mode_auto) pick
        # routing / stacking / flat_average PER DATASET. Slower (an internal k-fold comparison
        # runs before the real fit), but this is the most direct, already-implemented way to test
        # whether TRA's regression weak spot on regime-free data (see benchmark.py's documented
        # Friedman #2 / Synthetic Regression / Yacht results) is fixable with zero code changes.
        return {**common, "combination_mode": "auto"}
    raise ValueError(f"Unknown TRA variant: {variant!r}")


def make_tra_variant(task_type: str, variant: str):
    if OptimizedTRA is None:
        raise ImportError(
            f"Could not import tra_algorithm.core.OptimizedTRA: {TRA_IMPORT_ERROR}"
        )

    requested = get_tra_kwargs(task_type, variant)

    # Fail loudly (per-variant, reported in diagnostics) if the installed package predates a
    # requested parameter, rather than silently pretending every feature ran.
    signature = inspect.signature(OptimizedTRA)
    supported = set(signature.parameters)
    missing = [key for key in requested if key not in supported]
    kwargs = {k: v for k, v in requested.items() if k in supported}

    model = OptimizedTRA(**kwargs)
    model._benchmark_requested_tra_features = list(requested.keys())
    model._benchmark_unsupported_tra_features = missing
    return model


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

def _ensemble_base_estimators(task_type: str) -> List[Tuple[str, Any]]:
    """Base learners for the Voting/Stacking meta-ensembles: the strongest tree/boosting models
    available, matching what a well-resourced practitioner would actually combine, and matching
    the exact base-learner selection used in benchmark.py so both scripts' meta-ensemble rows are
    comparable. Falls back to sklearn's own GradientBoosting/ExtraTrees when XGBoost/LightGBM are
    not installed, so these two rows are never silently missing from the comparison.
    """
    if task_type == "classification":
        ests: List[Tuple[str, Any]] = [
            ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
        ]
        ests.append(("xgb", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE))
                    if HAS_XGBOOST else
                    ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE)))
        ests.append(("lgbm", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1))
                    if HAS_LIGHTGBM else
                    ("et", ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)))
        return ests

    ests = [("rf", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))]
    ests.append(("xgb", XGBRegressor(random_state=RANDOM_STATE))
                if HAS_XGBOOST else
                ("gb", GradientBoostingRegressor(random_state=RANDOM_STATE)))
    ests.append(("lgbm", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1))
                if HAS_LIGHTGBM else
                ("et", ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)))
    return ests


def get_models(task_type: str, tra_variants: List[str]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for variant in tra_variants:
        models[TRA_VARIANT_DISPLAY_NAMES[variant]] = make_tra_variant(task_type, variant)

    if task_type == "classification":
        models["LogisticRegression"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        )
        models["KNN"] = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        )
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        )
        models["ExtraTrees"] = ExtraTreesClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        )
        models["GradientBoosting"] = GradientBoostingClassifier(random_state=RANDOM_STATE)
        models["AdaBoost"] = AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)
        # SVC(probability=True) forces an internal 5-fold Platt-scaling CV and is deprecated as
        # of sklearn 1.9 (removed in 1.11). CalibratedClassifierCV(cv=3) is the supported, cheaper
        # replacement -- and is also literally what TRA's own SVM track builds internally.
        models["SVM"] = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE, probability=False), cv=3),
        )
        models["MLP"] = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
                         n_iter_no_change=15, random_state=RANDOM_STATE),
        )
        models["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=250, random_state=RANDOM_STATE,
        )

        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        if HAS_LIGHTGBM:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE,
                verbose=-1, n_jobs=-1,
            )
        if HAS_CATBOOST:
            models["CatBoost"] = CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05, verbose=False,
                random_state=RANDOM_STATE, thread_count=-1,
            )

        base = _ensemble_base_estimators("classification")
        models["Voting"] = VotingClassifier(estimators=base, voting="soft", n_jobs=1)
        models["Stacking"] = StackingClassifier(
            estimators=base, final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1,
        )

    else:
        models["Ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_STATE))
        models["KNN"] = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
        models["RandomForest"] = RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        )
        models["ExtraTrees"] = ExtraTreesRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        )
        models["GradientBoosting"] = GradientBoostingRegressor(random_state=RANDOM_STATE)
        models["AdaBoost"] = AdaBoostRegressor(n_estimators=100, random_state=RANDOM_STATE)
        models["SVR"] = make_pipeline(StandardScaler(), SVR())
        models["MLP"] = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
                        n_iter_no_change=15, random_state=RANDOM_STATE),
        )
        models["HistGradientBoosting"] = HistGradientBoostingRegressor(
            max_iter=250, random_state=RANDOM_STATE,
        )

        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        if HAS_LIGHTGBM:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE,
                verbose=-1, n_jobs=-1,
            )
        if HAS_CATBOOST:
            models["CatBoost"] = CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.05, verbose=False,
                random_state=RANDOM_STATE, thread_count=-1,
            )

        base = _ensemble_base_estimators("regression")
        models["Voting"] = VotingRegressor(estimators=base, n_jobs=1)
        models["Stacking"] = StackingRegressor(
            estimators=base, final_estimator=Ridge(), cv=3, n_jobs=1,
        )

    return models


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _as_proba_matrix(proba: np.ndarray) -> np.ndarray:
    """Normalise a predict_proba output to an (n, n_classes) matrix (some estimators return a
    1-D positive-class column for binary problems)."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return np.column_stack([1.0 - proba, proba])
    return proba


def _expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray,
                                n_bins: int = 15) -> float:
    """ECE of the top-label confidence, using EQUAL-MASS bins (equal-width bins are dominated by
    whichever bin holds most of the mass, which for an overconfident model is the top bin)."""
    proba = _as_proba_matrix(proba)
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    n = len(y_true)
    edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return float(abs(correct.mean() - conf.mean()))
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        m = int(in_bin.sum())
        if m:
            ece += (m / n) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """Multiclass Brier score (sklearn's brier_score_loss is BINARY only)."""
    proba = _as_proba_matrix(proba)
    idx = {c: i for i, c in enumerate(classes)}
    onehot = np.eye(len(classes))[[idx[v] for v in y_true]]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> Optional[float]:
    """ROC-AUC, binary or one-vs-rest multiclass as appropriate. sklearn's roc_auc_score requires
    a 1-D positive-class score for a genuinely binary problem (a 2-D 2-column array raises), and
    multi_class='ovr' only for 3+ classes -- this dispatches on n_classes so callers don't have to.
    """
    proba = _as_proba_matrix(proba)
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted",
                                   labels=classes))
    except Exception:
        return None


def evaluate_predictions(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if task_type == "classification":
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        if y_proba is not None and classes is not None:
            try:
                out["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
            except Exception:
                out["log_loss"] = np.nan
            try:
                out["roc_auc"] = _safe_roc_auc(y_true, y_proba, classes)
            except Exception:
                out["roc_auc"] = np.nan
            try:
                out["brier"] = _multiclass_brier(y_true, y_proba, classes)
            except Exception:
                out["brier"] = np.nan
            try:
                out["ece"] = _expected_calibration_error(y_true, y_proba, classes)
            except Exception:
                out["ece"] = np.nan
        else:
            out["log_loss"] = out["roc_auc"] = out["brier"] = out["ece"] = np.nan

        return out

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_proba_if_available(model, X) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# TRA diagnostics
# ---------------------------------------------------------------------------

def extract_tra_diagnostics(model) -> Dict[str, Any]:
    if not hasattr(model, "tracks"):
        return {}

    diagnostics: Dict[str, Any] = {}

    try:
        diagnostics["tra_tracks_after_fit"] = len(model.tracks)
    except Exception:
        pass

    for attr in [
        "_active_combination_mode_",
        "_dynamic_tracks_created",
        "_router_uses_meta_features",
        "_stacking_signals_active_",
        "routing_mode",
        "combination_mode",
        "router_use_oof",
        "difficulty_aware_fusion",
        "stacking_use_signals",
        "meta_learner_cv",
        "n_estimators",
    ]:
        try:
            value = getattr(model, attr)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            diagnostics[f"tra_{attr.strip('_')}"] = value
        except Exception:
            pass

    # Calibration report and diversity report are useful for a research-paper appendix.
    try:
        diagnostics["tra_calibration_report"] = model.get_calibration_report()
    except Exception:
        pass
    try:
        diagnostics["tra_diversity_report"] = model.get_diversity_report()
    except Exception:
        pass

    diagnostics["tra_unsupported_requested_features"] = getattr(
        model, "_benchmark_unsupported_tra_features", []
    )

    return diagnostics


# ---------------------------------------------------------------------------
# Primary static cross-validation benchmark
# ---------------------------------------------------------------------------

def run_cv_benchmark(
    dataset_names: List[str],
    folds: int,
    max_samples: int,
    tra_variants: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for d_i, dataset_name in enumerate(dataset_names, start=1):
        task_type, _ = DATASET_LOADERS[dataset_name]
        log("\n" + "=" * 90)
        log(f"[{d_i}/{len(dataset_names)}] DATASET: {dataset_name} | TASK: {task_type}")
        log("=" * 90)

        try:
            X, y, task_type = load_dataset(dataset_name, max_samples)
        except Exception as exc:
            error_rows.append({
                "Dataset": dataset_name, "Model": "__DATASET__", "Error": repr(exc),
            })
            log(f"Dataset failed: {exc}")
            continue

        classes = np.unique(y) if task_type == "classification" else None
        models = get_models(task_type, tra_variants)
        log(f"Samples={len(X):,} | Features={X.shape[1]:,} | Models={len(models)}")

        cv = (StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
              if task_type == "classification"
              else KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE))

        # Identical splits for all models -> per-fold scores are PAIRED (required for
        # compute_paired_significance below).
        splits = list(cv.split(X, y))

        for model_name, base_model in models.items():
            log(f"\n  [{model_name}]")

            for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
                model = clone(base_model)

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                start_fit = time.perf_counter()
                try:
                    model.fit(X_train, y_train)
                    fit_seconds = time.perf_counter() - start_fit

                    start_pred = time.perf_counter()
                    y_pred = np.asarray(model.predict(X_test))
                    predict_seconds = time.perf_counter() - start_pred

                    y_proba = (
                        get_proba_if_available(model, X_test)
                        if task_type == "classification"
                        else None
                    )

                    metrics = evaluate_predictions(
                        task_type, y_test, y_pred, y_proba, classes,
                    )

                    row = {
                        "Dataset": dataset_name,
                        "Task": task_type,
                        "Model": model_name,
                        "Fold": fold_id,
                        "Train_Size": len(train_idx),
                        "Test_Size": len(test_idx),
                        "Fit_Time_s": fit_seconds,
                        "Predict_Time_s": predict_seconds,
                        **metrics,
                    }

                    if model_name in TRA_VARIANT_DISPLAY_NAMES.values():
                        diag = extract_tra_diagnostics(model)
                        row["TRA_Tracks"] = diag.get("tra_tracks_after_fit")
                        row["TRA_Combination_Mode"] = diag.get("tra_active_combination_mode")
                        row["TRA_Unsupported_Features"] = json.dumps(
                            diag.get("tra_unsupported_requested_features", [])
                        )

                    fold_rows.append(row)

                    primary = (metrics["accuracy"] if task_type == "classification"
                              else metrics["rmse"])
                    log(f"    fold {fold_id}/{folds}: primary={primary:.5f}, "
                        f"fit={fit_seconds:.2f}s, predict={predict_seconds:.3f}s")

                except Exception as exc:
                    error_rows.append({
                        "Dataset": dataset_name, "Task": task_type, "Model": model_name,
                        "Fold": fold_id, "Error": repr(exc),
                    })
                    log(f"    fold {fold_id}/{folds}: FAILED -> {exc}")

        # Persist after every dataset so a long run that is interrupted never loses completed
        # work (the raw fold results are rewritten from the accumulator each time -- cheap
        # relative to the fitting cost).
        pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "benchmark_fold_results.csv", index=False)

    fold_df = pd.DataFrame(fold_rows)
    error_df = pd.DataFrame(error_rows)
    return fold_df, error_df


# ---------------------------------------------------------------------------
# Online/adaptation benchmark
# ---------------------------------------------------------------------------

def run_online_tra_benchmark(dataset_names: List[str], max_samples: int) -> pd.DataFrame:
    """
    Separate experiment for TRA's dynamic/online capabilities, using the "routing" variant
    (the simplest fusion mode to update incrementally -- "auto" would re-run its internal CV
    probe on every chunk, which is not a realistic streaming deployment).

    We intentionally do NOT mix this with the static CV leaderboard: the experiment
        1. fits TRA on an initial training block,
        2. evaluates sequential batches,
        3. calls partial_fit() with ground-truth labels (never pseudo-labels),
        4. records performance and expert-pool changes (dynamic spawning / Page-Hinkley drift).
    """
    rows: List[Dict[str, Any]] = []

    for dataset_name in dataset_names:
        task_type, _ = DATASET_LOADERS[dataset_name]

        try:
            X, y, task_type = load_dataset(dataset_name, max_samples)
        except Exception as exc:
            log(f"Online benchmark skipped for {dataset_name}: {exc}")
            continue

        if len(X) < 150:
            continue  # only meaningful for a dataset large enough to stream in batches

        rng = np.random.default_rng(RANDOM_STATE)
        order = rng.permutation(len(X))
        X, y = X[order], y[order]

        initial_size = max(80, int(len(X) * 0.45))
        remaining = len(X) - initial_size
        if remaining < 50:
            continue

        X_initial, y_initial = X[:initial_size], y[:initial_size]
        X_stream, y_stream = X[initial_size:], y[initial_size:]

        kwargs = get_tra_kwargs(task_type, "routing")
        kwargs.update({
            "enable_dynamic_spawning": True,
            "self_training": False,
            "max_dynamic_tracks": 3,
        })
        signature = inspect.signature(OptimizedTRA)
        kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

        try:
            model = OptimizedTRA(**kwargs)
            model.fit(X_initial, y_initial)
        except Exception as exc:
            rows.append({
                "Dataset": dataset_name, "Task": task_type,
                "Status": "FAILED_INITIAL_FIT", "Error": repr(exc),
            })
            continue

        classes = np.unique(y) if task_type == "classification" else None
        batch_size = max(25, min(100, remaining // 5))
        batch_id = 0

        for start in range(0, remaining, batch_size):
            end = min(start + batch_size, remaining)
            X_batch, y_batch = X_stream[start:end], y_stream[start:end]
            batch_id += 1

            try:
                before_tracks = len(getattr(model, "tracks", {}))

                start_pred = time.perf_counter()
                y_pred = np.asarray(model.predict(X_batch))
                predict_seconds = time.perf_counter() - start_pred

                metrics = evaluate_predictions(
                    task_type, y_batch, y_pred,
                    get_proba_if_available(model, X_batch) if task_type == "classification" else None,
                    classes,
                )

                start_update = time.perf_counter()
                if task_type == "classification":
                    model.partial_fit(X_batch, y_batch, classes=classes)
                else:
                    model.partial_fit(X_batch, y_batch)
                update_seconds = time.perf_counter() - start_update

                after_tracks = len(getattr(model, "tracks", {}))

                rows.append({
                    "Dataset": dataset_name, "Task": task_type, "Status": "OK",
                    "Batch": batch_id, "Batch_Size": len(X_batch),
                    "Predict_Time_s": predict_seconds, "Partial_Fit_Time_s": update_seconds,
                    "Tracks_Before": before_tracks, "Tracks_After": after_tracks,
                    "New_Tracks": after_tracks - before_tracks,
                    **metrics,
                })
            except Exception as exc:
                rows.append({
                    "Dataset": dataset_name, "Task": task_type,
                    "Status": "FAILED_BATCH", "Batch": batch_id, "Error": repr(exc),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation / ranking
# ---------------------------------------------------------------------------

METRIC_COLUMNS = [
    "accuracy", "f1_weighted", "f1_macro", "roc_auc", "log_loss", "brier", "ece",
    "rmse", "mae", "r2", "Fit_Time_s", "Predict_Time_s",
]


def aggregate_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return fold_df

    metric_columns = [c for c in METRIC_COLUMNS if c in fold_df.columns]

    metric_summary = (
        fold_df.groupby(["Dataset", "Task", "Model"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    metric_summary.columns = [
        "_".join(str(v) for v in col if str(v) != "") if isinstance(col, tuple) else str(col)
        for col in metric_summary.columns
    ]
    return metric_summary


def add_dataset_ranks(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    frames = []
    for dataset, subset in summary_df.groupby("Dataset", sort=False):
        subset = subset.copy()
        task = subset["Task"].iloc[0]

        if task == "classification":
            subset["Primary_Metric"] = subset["accuracy_mean"]
            subset["Primary_Metric_Name"] = "Accuracy"
            subset["Rank"] = subset["Primary_Metric"].rank(ascending=False, method="min").astype(int)
        else:
            subset["Primary_Metric"] = subset["rmse_mean"]
            subset["Primary_Metric_Name"] = "RMSE"
            subset["Rank"] = subset["Primary_Metric"].rank(ascending=True, method="min").astype(int)

        frames.append(subset)

    return pd.concat(frames, ignore_index=True)


def compute_average_ranks(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Demsar-style (2006) average rank per model, across all datasets within each task -- the
    standard way to summarize a multi-dataset comparison. A model averaging near 1 is
    consistently among the best; near the model count, consistently among the worst.
    """
    if summary_df.empty:
        return summary_df
    out = (
        summary_df.groupby(["Task", "Model"])["Rank"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Average_Rank", "count": "N_Datasets"})
        .sort_values(["Task", "Average_Rank"])
    )
    return out


def compute_paired_significance(fold_df: pd.DataFrame, tra_variants: List[str]) -> pd.DataFrame:
    """Paired t-test of every TRA variant against the strongest non-TRA baseline on each
    dataset's headline metric (accuracy for classification, RMSE for regression) -- valid because
    every model saw IDENTICAL folds (see run_cv_benchmark). Returns one row per
    (dataset, TRA variant) with the mean difference, 95% CI, p-value, and a BETTER/WORSE/ns
    verdict; a summary tally across all datasets is appended by write_markdown_report.
    """
    if fold_df.empty:
        return pd.DataFrame()

    tra_names = set(TRA_VARIANT_DISPLAY_NAMES[v] for v in tra_variants)
    rows = []

    for dataset, sub in fold_df.groupby("Dataset"):
        task = sub["Task"].iloc[0]
        metric, higher_better = ("accuracy", True) if task == "classification" else ("rmse", False)
        if metric not in sub.columns:
            continue

        means = sub.groupby("Model")[metric].mean()
        baseline_candidates = means.drop(labels=[n for n in tra_names if n in means.index],
                                         errors="ignore")
        if baseline_candidates.empty:
            continue
        ref_model = (baseline_candidates.idxmax() if higher_better else baseline_candidates.idxmin())

        ref_by_fold = (sub[sub["Model"] == ref_model]
                      .set_index("Fold")[metric].sort_index())

        for tra_name in tra_names:
            if tra_name not in sub["Model"].values:
                continue
            tra_by_fold = (sub[sub["Model"] == tra_name]
                          .set_index("Fold")[metric].sort_index())
            common_folds = ref_by_fold.index.intersection(tra_by_fold.index)
            if len(common_folds) < 2:
                continue

            a = tra_by_fold.loc[common_folds].to_numpy(dtype=float)
            b = ref_by_fold.loc[common_folds].to_numpy(dtype=float)
            d = a - b

            if np.allclose(d, d[0]):
                mean_diff, lo, hi, p = float(d.mean()), np.nan, np.nan, np.nan
            else:
                sem = scipy_stats.sem(d)
                half = sem * scipy_stats.t.ppf(0.975, len(d) - 1)
                mean_diff = float(d.mean())
                lo, hi = mean_diff - half, mean_diff + half
                p = float(scipy_stats.ttest_rel(a, b).pvalue)

            verdict = "ns"
            if np.isfinite(lo) and (lo > 0 or hi < 0):
                verdict = "BETTER" if ((mean_diff > 0) == higher_better) else "WORSE"

            rows.append({
                "Dataset": dataset, "Task": task, "Model": tra_name,
                "Baseline": ref_model, "Metric": metric,
                "Mean_Diff": mean_diff, "CI_Lo": lo, "CI_Hi": hi, "P_Value": p,
                "Verdict": verdict,
            })

    return pd.DataFrame(rows)


def summarize_significance_tally(significance_df: pd.DataFrame) -> pd.DataFrame:
    if significance_df.empty:
        return significance_df
    tally = (
        significance_df.groupby("Model")["Verdict"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["BETTER", "WORSE", "ns"], fill_value=0)
    )
    tally["N_Datasets"] = tally.sum(axis=1)
    return tally.reset_index()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    """DataFrame.to_markdown() requires the optional `tabulate` package. Fall back to a plain
    fixed-width text table (still readable, still pasteable into a paper draft) rather than
    crashing the whole run at the very last, purely cosmetic step after every fold already
    computed successfully."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def write_markdown_report(
    summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    error_df: pd.DataFrame,
    online_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    tra_variants: List[str],
    path: Path,
) -> None:
    lines: List[str] = []

    lines.append("# TRA Benchmark Report")
    lines.append("")
    lines.append(
        "Compares TRA (evidence-based configuration; see benchmark_full.py's module docstring) "
        "against popular ML baselines using identical cross-validation splits per dataset, "
        "including calibration metrics (log loss, Brier, ECE) and paired statistical significance."
    )
    lines.append("")
    lines.append(f"TRA variants tested: {', '.join(TRA_VARIANT_DISPLAY_NAMES[v] for v in tra_variants)}")
    lines.append("")

    if not summary_df.empty:
        lines.append("## Main leaderboard")
        lines.append("")
        cols = ["Dataset", "Task", "Model", "Primary_Metric_Name", "Primary_Metric", "Rank"]
        cols = [c for c in cols if c in summary_df.columns]
        leaderboard = summary_df[cols].sort_values(["Dataset", "Rank"]).round(5)
        lines.append(_to_markdown_safe(leaderboard, index=False))
        lines.append("")

    if not rank_df.empty:
        lines.append("## Average cross-dataset rank (Demsar 2006 style)")
        lines.append("")
        lines.append(
            "A model averaging near 1 is consistently among the best across the whole suite; "
            "near the model count, consistently among the worst."
        )
        lines.append("")
        lines.append(_to_markdown_safe(rank_df.round(2), index=False))
        lines.append("")

    if not significance_df.empty:
        lines.append("## Paired statistical significance (95% CI paired t-test)")
        lines.append("")
        lines.append(
            "Each TRA variant vs. the single strongest non-TRA baseline on that dataset's "
            "headline metric (accuracy for classification, RMSE for regression), using the "
            "identical CV folds every model saw. 'ns' = confidence interval spans zero."
        )
        lines.append("")
        tally = summarize_significance_tally(significance_df)
        lines.append("**Overall tally across all datasets:**")
        lines.append("")
        lines.append(_to_markdown_safe(tally, index=False))
        lines.append("")
        lines.append("**Per-dataset detail:**")
        lines.append("")
        sig_cols = ["Dataset", "Model", "Baseline", "Mean_Diff", "CI_Lo", "CI_Hi", "P_Value", "Verdict"]
        lines.append(_to_markdown_safe(significance_df[sig_cols].round(4), index=False))
        lines.append("")

    lines.append("## TRA configuration")
    lines.append("")
    lines.append(
        "TRA is evaluated at the settings validated on this project's own prior benchmarking "
        "(n_tracks=4, feature_selection=False, small_data_auto_mode=False, calibrate_output=False; "
        "everything else at core.py's own defaults -- see benchmark_full.py's module docstring for "
        "why every experimental flag from earlier versions of this script was removed rather than "
        "tuned further without evidence)."
    )
    lines.append("")

    if not online_df.empty:
        lines.append("## Online adaptation experiment")
        lines.append("")
        lines.append(
            "Evaluates labeled partial_fit adaptation (TRA (routing) variant) and tracks changes "
            "in the expert pool as new data streams in."
        )
        lines.append("")
        online_ok = online_df[online_df.get("Status") == "OK"] if "Status" in online_df.columns else online_df
        if not online_ok.empty:
            cols = ["Dataset", "Task", "Batch", "Batch_Size", "Tracks_Before", "Tracks_After",
                   "New_Tracks", "Predict_Time_s"]
            cols = [c for c in cols if c in online_ok.columns]
            lines.append(_to_markdown_safe(online_ok[cols].round(5), index=False))
            lines.append("")

    if not error_df.empty:
        lines.append("## Errors")
        lines.append("")
        lines.append(_to_markdown_safe(error_df, index=False))
        lines.append("")

    lines.append("## Interpretation guidance")
    lines.append("")
    lines.append("- Classification: higher Accuracy/F1/ROC-AUC is better; lower log loss/Brier/ECE is better.")
    lines.append("- Regression: lower RMSE/MAE is better; higher R2 is better.")
    lines.append(
        "- Log loss, Brier score, and ECE measure PROBABILITY CALIBRATION, not just top-1 "
        "correctness -- a model can be highly accurate yet poorly calibrated. This project's "
        "prior benchmarking found TRA's measurable, defensible advantage is calibration at "
        "accuracy parity, not raw accuracy superiority; check whether that pattern replicates "
        "here before writing a stronger claim than the paired-significance section supports."
    )
    lines.append(
        "- Runtime is reported separately and should not be folded into a single score unless "
        "the paper explicitly defines that utility function."
    )
    lines.append(
        "- A single benchmark win does not establish algorithmic superiority. Use the paired "
        "significance section and the average-rank table above, not any single dataset's row."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def create_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if summary_df.empty:
        return

    for dataset, subset in summary_df.groupby("Dataset"):
        task = subset["Task"].iloc[0]
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in dataset)

        if task == "classification":
            panels = [("accuracy_mean", "Mean Accuracy", True),
                     ("log_loss_mean", "Mean Log Loss (lower better)", False)]
        else:
            panels = [("rmse_mean", "Mean RMSE (lower better)", False),
                     ("r2_mean", "Mean R2", True)]

        panels = [(col, label, asc) for col, label, asc in panels if col in subset.columns]
        if not panels:
            continue

        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6))
        if len(panels) == 1:
            axes = [axes]

        for ax, (col, label, higher_is_better) in zip(axes, panels):
            ordered = subset.sort_values(col, ascending=higher_is_better)
            ax.barh(ordered["Model"], ordered[col])
            ax.set_xlabel(label)
            ax.set_title(f"{dataset} — {label}")

        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_name}_leaderboard.png", dpi=160)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark TRA against popular ML algorithms (research-paper suite)."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_LOADERS.keys()),
        choices=list(DATASET_LOADERS.keys()), help="Datasets to benchmark.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument(
        "--max-samples", type=int, default=5000,
        help="Maximum samples per dataset. 0 = no cap.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast sanity check: <=300 rows/dataset, 3-fold CV. Overrides --folds/--max-samples.",
    )
    parser.add_argument(
        "--include-difficulty-aware", action="store_true",
        help="Add a 4th TRA variant, 'TRA (routing, difficulty-aware)' (difficulty_aware_fusion=True).",
    )
    parser.add_argument("--skip-online", action="store_true", help="Skip the dynamic/online TRA experiment.")
    parser.add_argument("--skip-plots", action="store_true", help="Do not generate PNG plots.")
    parser.add_argument("--output-dir", default="benchmark_outputs", help="Directory for benchmark artifacts.")
    return parser.parse_args()


def main():
    args = parse_args()

    folds = 3 if args.quick else args.folds
    max_samples = 300 if args.quick else args.max_samples

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_STATE)

    tra_variants = ["stacking", "routing", "auto"]
    if args.include_difficulty_aware:
        tra_variants.append("routing-difficulty-aware")

    log("=" * 90)
    log("TRA FULL-FEATURE BENCHMARK (research-paper suite)")
    log("=" * 90)
    log(f"Output directory: {OUTPUT_DIR.resolve()}")
    log(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    log(f"Datasets ({len(args.datasets)}): {', '.join(args.datasets)}")
    log(f"CV folds: {folds}")
    log(f"Max samples/dataset: {max_samples or 'unlimited'}")
    log(f"TRA variants: {', '.join(TRA_VARIANT_DISPLAY_NAMES[v] for v in tra_variants)}")
    log("")

    if OptimizedTRA is None:
        log("ERROR: tra_algorithm.core.OptimizedTRA could not be imported.")
        log(f"Import error: {TRA_IMPORT_ERROR}")
        sys.exit(1)

    log("Optional libraries:")
    log(f"  XGBoost : {HAS_XGBOOST}")
    log(f"  LightGBM: {HAS_LIGHTGBM}")
    log(f"  CatBoost: {HAS_CATBOOST}")
    log("")

    # Save the exact TRA configuration used by this run, per variant, for reproducibility.
    config_path = OUTPUT_DIR / "tra_full_configuration.json"
    config = {
        variant: get_tra_kwargs(DATASET_LOADERS[args.datasets[0]][0], variant)
        for variant in tra_variants
    }
    config_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    t_start = time.time()
    fold_df, error_df = run_cv_benchmark(args.datasets, folds, max_samples, tra_variants)
    total_minutes = (time.time() - t_start) / 60.0

    if fold_df.empty:
        log("\nNo successful benchmark folds were produced.")
        error_df.to_csv(OUTPUT_DIR / "benchmark_errors.csv", index=False)
        sys.exit(2)

    # Main_Metric column for cross-tool compatibility with statistical_analysis.py's Friedman
    # test / average-rank pipeline (accuracy for classification rows, rmse for regression rows).
    fold_df["Main_Metric"] = np.where(
        fold_df["Task"] == "classification", fold_df["accuracy"], fold_df["rmse"]
    )

    summary_df = aggregate_results(fold_df)
    summary_df = add_dataset_ranks(summary_df)
    rank_df = compute_average_ranks(summary_df)
    significance_df = compute_paired_significance(fold_df, tra_variants)

    fold_df.to_csv(OUTPUT_DIR / "benchmark_fold_results.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)
    error_df.to_csv(OUTPUT_DIR / "benchmark_errors.csv", index=False)
    rank_df.to_csv(OUTPUT_DIR / "average_ranks.csv", index=False)
    significance_df.to_csv(OUTPUT_DIR / "significance_results.csv", index=False)

    online_df = pd.DataFrame()
    if not args.skip_online:
        log("\n" + "=" * 90)
        log("TRA ONLINE / DYNAMIC ADAPTATION EXPERIMENT")
        log("=" * 90)
        online_df = run_online_tra_benchmark(args.datasets, max_samples)
        online_df.to_csv(OUTPUT_DIR / "tra_online_results.csv", index=False)

    write_markdown_report(
        summary_df, fold_df, error_df, online_df, significance_df, rank_df, tra_variants,
        OUTPUT_DIR / "benchmark_report.md",
    )

    if not args.skip_plots:
        create_plots(summary_df, OUTPUT_DIR)

    log("\n" + "=" * 90)
    log("FINAL LEADERBOARD")
    log("=" * 90)
    display_cols = ["Dataset", "Task", "Model", "Primary_Metric_Name", "Primary_Metric", "Rank"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].sort_values(["Dataset", "Rank"]).round(5).to_string(index=False))

    if not rank_df.empty:
        log("\n" + "=" * 90)
        log("AVERAGE CROSS-DATASET RANK")
        log("=" * 90)
        print(rank_df.round(2).to_string(index=False))

    if not significance_df.empty:
        log("\n" + "=" * 90)
        log("PAIRED SIGNIFICANCE TALLY (headline metric, vs. strongest baseline per dataset)")
        log("=" * 90)
        print(summarize_significance_tally(significance_df).to_string(index=False))

    log(f"\nTotal time: {total_minutes:.1f} min")
    log("Benchmark complete.")
    log(f"Results: {OUTPUT_DIR.resolve()}")
    log("  - benchmark_fold_results.csv   (per-fold, per-model raw metrics + Main_Metric)")
    log("  - benchmark_summary.csv        (per-dataset mean/std + rank)")
    log("  - average_ranks.csv            (Demsar-style average rank per model)")
    log("  - significance_results.csv     (paired t-test, TRA vs. strongest baseline)")
    log("  - benchmark_errors.csv")
    log("  - tra_online_results.csv       (unless --skip-online)")
    log("  - tra_full_configuration.json")
    log("  - benchmark_report.md")
    log("  - leaderboard PNGs             (unless --skip-plots)")


if __name__ == "__main__":
    main()
