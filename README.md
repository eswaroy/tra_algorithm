# TRA: Track/Rail Algorithm.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](pyproject.toml)

TRA (Track/Rail Algorithm) is a signal-guided Mixture-of-Experts (MoE) architecture for
classification and regression on tabular data. It combines a pool of heterogeneous expert
models ("tracks") with a routing or stacking layer that decides how to combine their outputs,
guided by structural signals about each input's difficulty rather than raw features alone.

The implementation lives in `tra_algorithm/core.py` as the class `EnhancedTRA` (exported under
the alias `OptimizedTRA` for backward compatibility; the two names refer to the exact same
class). This document describes the current, validated state of the algorithm, its
architecture, its configuration, and the results of a 20-dataset benchmark against 13 widely
used machine learning algorithms.

## Table of Contents

- [What TRA Is](#what-tra-is)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Recommended Configuration](#recommended-configuration)
- [Architecture](#architecture)
- [Parameter Reference](#parameter-reference)
- [Advanced Capabilities](#advanced-capabilities)
- [Empirical Validation](#empirical-validation)
- [When to Use TRA](#when-to-use-tra)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)

## What TRA Is

Most machine learning models assume a single function can describe the relationship between
inputs and outputs across an entire dataset. Real data frequently contains latent
sub-populations ("regimes") that behave differently from one another. TRA's design responds to
this by training several specialist models and combining their outputs through a learned
routing or stacking layer, rather than fitting one model to the whole distribution at once.

TRA's specific contribution, relative to a generic Mixture-of-Experts system, is **signal-guided
routing**: before a routing or fusion decision is made, TRA computes five structural signals
describing how difficult or unusual an input looks, and factors these into the decision
alongside the raw features:

1. **Expert disagreement** -- the standard deviation of the tracks' raw predictions for this
   input.
2. **Prediction entropy** -- the entropy of the ensemble's mean class-probability distribution
   (classification only).
3. **Feature density** -- distance to the nearest neighbors in feature space (is this input in a
   well-populated or a sparse region of the training data).
4. **Cluster distance** -- distance to the nearest centroid of a KMeans clustering fitted on the
   training data.
5. **Outlier score** -- an Isolation Forest anomaly score.

These signals are computed by the `SignalExtractor` class and are consumed by both of TRA's
fusion strategies: the router (when `combination_mode="routing"`) and the stacking meta-learner
(when `combination_mode="stacking"`, via `stacking_use_signals=True`, the default).

## Installation

TRA is published on PyPI:

```bash
pip install tra-algorithm
```

```python
from tra_algorithm import EnhancedTRA  # or the alias OptimizedTRA
```

### Installing from source

For development, or to use unreleased changes from this repository:

```bash
git clone https://github.com/eswaroy/tra_algorithm.git
cd tra_algorithm
python -m venv venv
```

Activate the virtual environment, then install dependencies:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Linux / macOS
source venv/bin/activate
pip install -r requirements.txt
```

For an editable package install (uses `setup.py` / `pyproject.toml`):

```bash
pip install -e .
```

### Optional gradient-boosting backends

TRA's default expert pool and router use XGBoost and LightGBM when available. Install them for
the full expert pool and the strongest available router:

```bash
pip install xgboost lightgbm catboost
```

Each is optional. If any is missing, `core.py` degrades gracefully: the expert pool and router
fall back to the next-strongest available option (ultimately RandomForest, which has no optional
dependency), and a message is logged noting what was skipped. TRA does not raise an import error
for a missing optional library.

## Quick Start

### Classification

```python
from tra_algorithm import EnhancedTRA
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = EnhancedTRA(
    task_type="classification",
    n_tracks=4,
    combination_mode="stacking",
    small_data_auto_mode=False,
    feature_selection=False,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'):.4f}")
```

### Regression

```python
from tra_algorithm import EnhancedTRA
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = EnhancedTRA(
    task_type="regression",
    n_tracks=4,
    combination_mode="routing",
    router_use_oof=True,
    small_data_auto_mode=False,
    feature_selection=False,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R2:   {r2_score(y_test, y_pred):.4f}")
```

`OptimizedTRA` is the same class as `EnhancedTRA` and both import paths work identically;
`EnhancedTRA` is used above because it is the name the class itself uses internally.

## Recommended Configuration

The two configurations below are the ones actually validated by this project's own benchmark
(see [Empirical Validation](#empirical-validation)); they are not a guess at reasonable defaults.

```python
# Classification -- best calibration and competitive accuracy in this project's benchmark.
EnhancedTRA(
    task_type="classification",
    n_tracks=4,
    combination_mode="stacking",
    small_data_auto_mode=False,
    feature_selection=False,
    random_state=42,
)

# Regression -- best of the two fusion modes on this project's benchmark, though TRA is
# not currently competitive on regression tasks without latent regime structure (see below).
EnhancedTRA(
    task_type="regression",
    n_tracks=4,
    combination_mode="routing",
    router_use_oof=True,
    small_data_auto_mode=False,
    feature_selection=False,
    random_state=42,
)
```

Why these specific settings, and not others that appear in the constructor:

- **`n_tracks=4`** produces a pool of RandomForest, LightGBM, XGBoost, and SVM experts (in that
  order, when all optional libraries are installed) -- four genuinely different model families,
  which is what gives an ensemble diversity to exploit. It also means TRA's expert pool contains
  the same model families it is typically benchmarked against.
- **`feature_selection=False`** is a hard requirement, not a stylistic choice. When enabled, this
  option applies a univariate feature filter that ranks features by their global correlation
  with the target. That ranking is the wrong criterion for a mixture-of-experts: a feature can be
  strongly predictive within one regime while its effect cancels out globally, scoring near zero
  and getting dropped. `False` is also core.py's own default.
- **`small_data_auto_mode=False`** is required whenever `combination_mode` is set explicitly.
  Left at its default (`True`), TRA silently overrides `combination_mode="routing"` to
  `"stacking"` on any dataset under 2000 rows, which is most datasets -- meaning both
  configurations above would silently become the same model unless this is turned off.
- **`combination_mode`** differs between the two configurations because the two fusion strategies
  are not equally good at both tasks in this project's measurements: stacking measurably wins on
  classification calibration, and routing (with `router_use_oof=True`, which trains the router on
  leak-free out-of-fold labels across the whole training set instead of a single holdout split)
  is the stronger of the two regression configurations, though neither is currently competitive
  with the strongest gradient-boosting baselines on regression (see below).

Every other constructor parameter is left at its current default rather than being overridden
here. As of this version those defaults already include `n_estimators=100`, `meta_learner_cv=True`
(with the meta-learner's regularization strength selected by log loss, not accuracy, so it
optimizes calibration rather than working against it), and `stacking_use_signals=True` (the five
structural signals above are concatenated onto the stacking meta-learner's input, not only used
by the router) -- all three were validated on the full benchmark suite to improve accuracy,
ROC-AUC, log loss, and calibration with no measured regression.

## Architecture

```text
Input
  -> Preprocessing (imputation, scaling; categorical encoding for DataFrame input)
  -> Optional feature selection (off by default; see above)
  -> Expert track creation (heterogeneous models: RandomForest, LightGBM, XGBoost, SVM, ...)
  -> Signal extraction layer (5 structural signals, fitted on the training data)
  -> Leak-free out-of-fold evaluation of every track (k-fold, shared across all consumers below)
  -> Fusion layer: ONE of
       - routing:       a trained router (XGBoost by default) predicts which track (or weighted
                         blend of tracks, under the default soft routing mode) to use per sample
       - stacking:      a meta-learner (LogisticRegression / Ridge, cross-validated regularization
                         by default) is fit on every track's out-of-fold output plus the 5
                         structural signals
       - flat_average:  a uniform average over every track, with no router and no meta-learner
       - auto:          an internal cross-validated comparison among the three modes above picks
                         the winner for this dataset
  -> Optional residual correction track ("TRA-Boost"): a specialist trained on the fused
     ensemble's out-of-fold errors, kept only if it demonstrates a held-out accuracy/error
     improvement above a configurable threshold; discarded otherwise
  -> Optional output calibration (classification) / conformal calibration (regression)
  -> Final prediction
```

### Fusion strategies in detail

- **Routing** trains a classifier (the "router") whose target is "which track performs best on
  this input," using leak-free out-of-fold labels when `router_use_oof=True`. At inference, the
  default soft routing mode (`routing_mode="soft"`) blends every track's output, weighted by the
  router's full predicted probability distribution over tracks -- it does not restrict itself to
  a fixed top-K subset by default. The `top_k` parameter only has an effect when
  `intelligent_routing=True` (an experimental hierarchical-routing subsystem, off by default and
  not part of the validated configuration above); under ordinary soft routing, `top_k` is unused.
- **Stacking** fits a meta-learner directly on every track's out-of-fold prediction (probabilities
  for classification, point predictions for regression), plus the five structural signals when
  `stacking_use_signals=True`. This is the mode that produced this project's best-calibrated
  results (see below).
- **Flat averaging** exists as a deliberate, simple fallback: on data with no real regime
  structure, an unweighted average across experts can outperform a router or meta-learner that is
  fitting noise. `combination_mode="auto"` includes this as a candidate.
- **Correction track**: after fusion, TRA checks whether there is enough signal in the fused
  ensemble's held-out mistakes to train a useful "cleanup" specialist. The check is itself
  validated on a held-out split before deployment; if the projected gain does not clear
  `correction_gain_threshold`, the correction track is discarded rather than shipped inactive.

### Key classes

| Class | Role |
|---|---|
| `EnhancedTRA` (alias `OptimizedTRA`) | Main estimator; implements the scikit-learn `fit`/`predict`/`predict_proba` interface |
| `SignalExtractor` | Computes the five structural signals described above |
| `Track` | Wraps one expert model with usage/performance bookkeeping |
| `PageHinkleyDetector` | Lightweight concept-drift test used by `partial_fit` |

## Parameter Reference

This lists the constructor parameters most relevant to day-to-day use, with their actual current
defaults (all values below were read directly from `tra_algorithm/core.py`). The full constructor
has additional parameters governing experimental subsystems (dynamic track spawning, hierarchical
"intelligent" routing, loss-aware routing weighting); those are intentionally omitted here because
they are not part of the validated configuration and are documented in the source directly.

### Core

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task_type` | str | `"classification"` | `"classification"` or `"regression"` |
| `n_tracks` | int | `5` | Number of expert tracks (use `4` for the validated configuration) |
| `max_tracks` | int | `8` | Upper bound on total tracks, including any dynamically spawned |
| `router_type` | str | `"xgboost"` | Router backend: `"xgboost"`, `"catboost"`, `"lightgbm"`, `"mlp"` |
| `routing_mode` | str | `"soft"` | `"soft"` (weighted blend) or `"hard"` (single selected track) |
| `combination_mode` | str | `"routing"` | `"routing"`, `"stacking"`, `"flat_average"`, or `"auto"` |
| `random_state` | int or None | `None` | Seed for reproducibility |
| `n_jobs` | int | `-1` | Parallelism for tree-based tracks/router |

### Fusion-related

| Parameter | Type | Default | Description |
|---|---|---|---|
| `router_use_oof` | bool | `False` | Train the router on leak-free out-of-fold labels across the full training set instead of a single holdout split |
| `stacking_folds` | int | `5` | Fold count for the out-of-fold pass consumed by stacking/routing/the correction track |
| `stacking_use_signals` | bool | `True` | Concatenate the 5 structural signals onto the stacking meta-learner's input |
| `meta_learner_cv` | bool | `True` | Use `LogisticRegressionCV`/`RidgeCV` (log-loss-scored for classification) instead of fixed regularization for the stacking meta-learner |
| `difficulty_aware_fusion` | bool | `False` | Blend the routed/stacked decision toward a flat average in proportion to a per-sample predicted-difficulty score |
| `small_data_auto_mode` | bool | `True` | Auto-switch to data-efficient settings under `small_data_threshold` rows; set `False` to keep an explicitly chosen `combination_mode` |
| `small_data_threshold` | int | `2000` | Row-count threshold for the above |

### Expert tracks

| Parameter | Type | Default | Description |
|---|---|---|---|
| `track_models` | list or None | `None` | Supply custom expert models instead of the built-in heterogeneous pool |
| `feature_selection` | bool | `False` | Enable a feature-selection pre-filter (see [Recommended Configuration](#recommended-configuration) for why this defaults off) |
| `feature_selection_method` | str | `"model_based"` | `"model_based"`, `"mutual_info"`, or `"univariate"` |
| `n_estimators` | int | `100` | Tree count for RandomForest/LightGBM/XGBoost/CatBoost tracks |
| `max_depth` | int | `6` | Max tree depth for tree-based tracks |
| `bagging_mode` | str | `"full"` | `"full"` (every track sees the complete training set) or `"bootstrap"` (per-track row bootstrap) |
| `cluster_experts` | bool | `False` | Assign training rows to tracks via KMeans clustering instead of the bagging mode above |
| `calibrate_tracks` | bool | `False` | Wrap each track's classifier in `CalibratedClassifierCV` before fusion |

### Correction track, calibration, and uncertainty

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable_correction_track` | bool | `True` | Train the TRA-Boost residual correction specialist (see above; self-gating) |
| `correction_gain_threshold` | float | `0.02` | Minimum validated held-out gain required to keep the correction track |
| `calibrate_output` | bool | `False` | Fit ensemble-level probability calibration on a held-out slice (classification). Measured on this project's benchmark as reducing accuracy without improving ECE; left off in the validated configuration |
| `enable_conformal` | bool | `True` | Fit split-conformal regression intervals (used by `predict_interval`) |
| `enable_difficulty_model` | bool | `True` | Fit the auxiliary difficulty estimator used by `predict_difficulty()` and (if enabled) `difficulty_aware_fusion` |

### Other

| Parameter | Type | Default | Description |
|---|---|---|---|
| `handle_imbalanced` | bool | `True` | Compute balanced class weights for classification |
| `abstention_threshold` | float | `0.0` | Abstain (return a sentinel) when routing confidence is below this value |
| `enable_dynamic_spawning` | bool | `False` | Allow new specialist tracks to be created from `partial_fit` data (see [Advanced Capabilities](#advanced-capabilities)) |
| `max_workers` | int | `4` | Worker threads for parallel track fitting (capped at 8) |

## Advanced Capabilities

### Prediction intervals (regression)

```python
point_pred, lower, upper = model.predict_interval(X_test, confidence=0.90)
```

When `enable_conformal=True` (the default) and enough data was available at fit time, this uses
split-conformal calibration on a held-out slice that was carved off before any track, router, or
stacking training happened -- under the standard conformal exchangeability assumption this gives
approximately valid marginal coverage. It is not a guarantee under concept drift or with a small,
unrepresentative calibration slice, and TRA logs a warning and falls back to an explicitly
uncalibrated ensemble-disagreement interval when conformal calibration was not possible.

### Probability calibration report

```python
model = EnhancedTRA(task_type="classification", calibrate_output=True, ...)
model.fit(X_train, y_train)
report = model.get_calibration_report()
```

When `calibrate_output=True`, TRA tries several calibration methods (temperature scaling,
sigmoid, isotonic) against a "do nothing" baseline on a genuinely held-out slice, and keeps
whichever has the lowest held-out log loss -- "none" is a real candidate, so calibration cannot
report itself as an improvement unless it measurably is one. `get_calibration_report()` returns
the metrics for every candidate that was tried.

### Per-sample difficulty estimation

```python
difficulty = model.predict_difficulty(X_test)  # values in [0, 1]
```

Available when `enable_difficulty_model=True` (the default). This regresses the five structural
signals against genuine out-of-fold ensemble loss at training time, so it is usable at inference
without needing labels. It is a diagnostic proxy for expected ensemble error, not a calibrated
probability of misclassification.

### Online learning and concept drift

```python
model.partial_fit(X_batch, y_batch)          # regression
model.partial_fit(X_batch, y_batch, classes=all_classes)  # classification, first call
```

`partial_fit` trains new tracks on the incoming batch and retrains the fusion layer against the
current track pool, using ground-truth labels only (never the model's own predictions, unless
`self_training=True` is explicitly opted into). A built-in Page-Hinkley test monitors per-batch
loss and can trigger drift-aware behavior. Track pruning (`enable_track_pruning=True`, the
default) periodically removes tracks that are both rarely selected and, per a leak-free
out-of-fold ablation, contribute little to ensemble accuracy.

### Model persistence

```python
model.save_model("model.joblib")
loaded = EnhancedTRA.load_model("model.joblib")
```

`save_model` stores research-grade metadata (library versions, architecture version, feature
schema) alongside the pickled estimator so a saved file can be partially inspected without
unpickling the full model, and so `load_model` can warn on an architecture-version mismatch.

## Empirical Validation

TRA was benchmarked against 13 widely used algorithms -- Logistic/Ridge regression, KNN,
RandomForest, ExtraTrees, GradientBoosting, AdaBoost, SVM/SVR, an MLP, XGBoost, LightGBM,
CatBoost, and hand-built Voting and Stacking meta-ensembles over the strongest available
tree/boosting models -- across 20 datasets (10 classification, 10 regression: Breast Cancer,
Wine, Digits, Ionosphere, Sonar, Pima Indians Diabetes, Vehicle Silhouettes, Banknote
Authentication, Spambase, Glass Identification; Diabetes, California Housing, Friedman #1/#2/#3,
Wine Quality Red, Concrete Compressive Strength, Auto MPG, Yacht Hydrodynamics, and a synthetic
high-dimensional linear regression set). Every model was evaluated on identical 5-fold
cross-validation splits per dataset, which makes a paired t-test valid; results below use that
pairing. The full benchmark is reproducible via `benchmark.py` and `benchmark_full.py` in this
repository.

### Classification: competitive accuracy, best calibration

Averaged rank across the 10 classification datasets, out of 15 models compared (1 = best):

| Metric | TRA (stacking) rank | TRA (routing) rank | Best-ranked model |
|---|---|---|---|
| Accuracy | 5.05 | 6.00 | ExtraTrees (3.80) |
| ROC-AUC | 4.60 | 5.40 | CatBoost (3.80) |
| Log loss | **2.50 (best of 15)** | 4.70 | -- |
| Expected Calibration Error | 5.20 | 4.90 | KNN (4.60) |

Mean accuracy across the 10 classification datasets: TRA (stacking) 89.78%, versus 90.20% for the
top-ranked model (ExtraTrees) -- a gap of roughly four-tenths of one percentage point. TRA
(stacking) has the best average log-loss rank of any of the 15 models tested, including every
gradient-boosting library in the comparison. Head-to-head against a Stacking ensemble built from
the identical underlying model families TRA's own tracks use, TRA is accuracy-neutral but wins on
log loss and ECE on the majority of datasets -- indicating the calibration advantage comes from
TRA's signal-guided fusion mechanism specifically, not merely from stacking in general.

### Regression: not currently competitive without regime structure

Averaged RMSE rank across the 10 regression datasets: TRA (stacking) 7.1, TRA (routing) 7.4, out
of 15 models (CatBoost ranks best at 3.9). On three of the ten datasets, TRA's error is
substantially worse than the best available model:

| Dataset | TRA RMSE vs. best model |
|---|---|
| Friedman #2 | 9.9x -- 10.5x worse |
| Synthetic Regression (linear-generated) | 8.2x -- 9.3x worse |
| Yacht Hydrodynamics | 5.2x -- 5.6x worse |

TRA's architecture assumes the data contains latent regimes that separate experts can each own.
None of these three datasets have that structure -- the synthetic regression set is generated by
a purely linear function, which a plain linear model wins outright. On data without regime
structure, routing or stacking across experts adds variance with nothing to offset it.
`combination_mode="auto"` (which includes flat averaging as a candidate) is the most direct,
already-implemented way to test whether this is recoverable on a given dataset without any
architecture change; it has not yet been validated in aggregate across this benchmark suite.

### Statistical significance and cost

Using a paired t-test against the single strongest available baseline on each dataset's headline
metric (accuracy for classification, RMSE for regression), at 95% confidence, across all 20
datasets:

| Model | Significantly better | Significantly worse | Not distinguishable |
|---|---|---|---|
| TRA (stacking) | 0 / 20 | 9 / 20 | 11 / 20 |
| TRA (routing) | 0 / 20 | 8 / 20 | 12 / 20 |

TRA does not achieve a statistically significant win over the best available baseline's raw
accuracy or RMSE on any of the 20 datasets tested; it is statistically indistinguishable from the
best baseline on roughly half. TRA is also markedly slower to train: averaged across all 20
datasets, TRA (stacking) and TRA (routing) took approximately 27-29 seconds per cross-validation
fold, versus under 1 second for LightGBM and under 7 seconds for every other baseline except
CatBoost (about 21s) and a from-scratch Stacking ensemble (about 22s), which are in a similar
cost range to TRA itself.

### Summary

The evidence supports a specific, narrower claim than "TRA outperforms existing methods": **TRA
reaches classification accuracy statistically indistinguishable from the strongest available
baselines while producing the best-calibrated probability estimates of any model in this
benchmark, at a real computational cost, and is not currently competitive for regression tasks on
data without latent regime structure.**

## When to Use TRA

**Reasonable fit:**
- Classification problems where calibrated probabilities matter as much as, or more than, raw
  top-1 accuracy -- risk scoring, triage support, or any pipeline that acts on the predicted
  probability itself rather than only the predicted class.
- Situations where a 10-30x training-time cost relative to XGBoost/LightGBM is acceptable.
- Data plausibly containing distinct sub-populations or operating regimes.

**Poor fit currently:**
- Regression tasks, particularly on smooth, homogeneous data without known sub-populations.
- Latency- or resource-constrained training pipelines.
- Any use case whose success criterion is "highest possible raw accuracy or RMSE" -- the
  benchmark evidence does not support a claim of outright superiority on that criterion.

## Project Structure

```text
tra_algorithm/
    core.py          EnhancedTRA (alias OptimizedTRA), SignalExtractor, Track, PageHinkleyDetector
    utils.py         Dataset generation and evaluation helpers
    examples.py       Worked classification/regression/comparison examples
    __init__.py       Public package exports
tests/
    test_core.py      Unit tests for EnhancedTRA
    test_utils.py      Unit tests for the utils module
benchmark.py          20-dataset benchmark against 15 models, with paired significance testing
benchmark_full.py     Extended benchmark suite: 3 TRA fusion variants, 17 models, calibration
                      metrics, average-rank tables, and a separate online/drift-adaptation experiment
statistical_analysis.py  Friedman test and average-rank tables over benchmark.py's output
requirements.txt      Runtime and optional dependencies
setup.py / pyproject.toml  Package metadata
```

## Requirements

```text
python >= 3.8
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.3.0
joblib >= 1.0.0
networkx >= 2.6.0
scipy >= 1.7.0
```

Optional, for the full expert pool, router options, and benchmark scripts:

```text
xgboost >= 2.0.0
lightgbm >= 4.0.0
catboost >= 1.2
tabulate >= 0.9.0   (benchmark_full.py's Markdown report tables; falls back to plain text if absent)
```

See `requirements.txt` for the complete, versioned list, including development dependencies
(`pytest`, `black`, `flake8`, `mypy`, `sphinx`).

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Citation

```bibtex
@software{tra_algorithm,
  title = {TRA: Track/Rail Algorithm -- Signal-Guided Mixture-of-Experts for Classification and Regression},
  author = {Ranga Eswar, Dasari},
  url = {https://github.com/eswaroy/tra_algorithm},
  license = {MIT}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.
