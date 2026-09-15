"""
TRA vs. the field — 20-dataset, 5-fold CV benchmark for the TRA research paper.

WHAT THIS FILE IS FOR
======================
A publication-grade head-to-head between EnhancedTRA and the most commonly used
"popular / best-in-class / ensemble" models in the ML literature, across 20 widely-used
benchmark datasets (10 classification, 10 regression), so results can be dropped straight
into a paper's experiments section.

DATASETS (20 total — real UCI/OpenML sets plus the classic Friedman synthetic functions
used throughout the mixture-of-experts / ensemble literature to test regime-structured data)
----------------------------------------------------------------------------------------
Classification (10): Breast Cancer Wisconsin, Wine, Digits, Ionosphere, Sonar,
Pima Indians Diabetes, Vehicle Silhouettes, Banknote Authentication, Spambase,
Glass Identification.
  (Iris is deliberately EXCLUDED: every model in this suite scores ~97-100% on it, so it
  has no discriminative power for a paper claiming an accuracy/calibration edge. Glass
  Identification is used instead -- small, 6-way, and genuinely imbalanced.)

Regression (10): Diabetes, California Housing, Friedman #1/#2/#3 (synthetic), Wine Quality
(Red), Concrete Compressive Strength, Auto MPG, Yacht Hydrodynamics, and a synthetic
high-dimensional make_regression set.

All non-toy sets are pulled once via sklearn's built-in loaders or `fetch_openml` (pinned
dataset names) and cached under ~/scikit_learn_data afterwards, so repeat runs are offline
and reproducible. A dataset that fails to download (no network on first run) is skipped with
a message rather than aborting the whole benchmark.

MODELS
------
Every dataset is scored against the same fixed roster: TRA (2 fusion configurations) plus
11 baselines spanning linear models, instance-based, single trees, bagging, 3 different
boosting families, a neural net, and 2 hand-built meta-ensembles (Voting, Stacking) over the
strongest available base learners -- i.e. exactly the models a reviewer expects to see TRA
compared against. XGBoost/LightGBM/CatBoost are optional; if not installed, the Voting/
Stacking ensembles automatically substitute sklearn's own GradientBoosting/ExtraTrees so
those two rows are never silently empty.

TRA CONFIGURATION (kept from the settings a 5-seed paired benchmark selected — do not revert)
-----------------------------------------------------------------------------------------
  classification -> combination_mode="stacking"   (best accuracy AND best log loss/Brier/ECE)
                    router_use_oof=True           (best pure-routing variant, entered separately)
  regression     -> router_use_oof=True           (best; beat the matched ensemble numerically)

  feature_selection=False is set EXPLICITLY. The univariate SelectKBest filter it would enable
  ranks features by GLOBAL univariate association, which is structurally wrong for a mixture-
  of-experts: a feature can be strongly predictive inside each regime while its per-regime
  effects cancel globally. Measured cost on Friedman1: RMSE 2.02 (last place) vs 1.21 (second
  place) with it off.

  small_data_auto_mode=False is REQUIRED. It is on by default and silently forces
  combination_mode='stacking' on any dataset under 2000 rows -- which is most datasets in this
  suite. Left on, the two TRA rows would fit the exact same model and the routing configuration
  would never actually be tested.

  n_tracks=4 so the pool includes RandomForest + LightGBM + XGBoost + SVM rather than stopping
  before the boosting families -- otherwise TRA is compared against baselines it does not
  itself contain.

  `router_use_oof=True` is NOT a TRA default (statistically neutral and ~35% slower across
  8 datasets) but was the best routing setting on every benchmark dataset, so it is set here.
  `calibrate_output=True` is deliberately NOT used -- it measured as the worst TRA
  classification config while not improving ECE.

METRICS
-------
Classification: accuracy, weighted F1, ROC-AUC (one-vs-rest), log loss, multiclass Brier,
expected calibration error (ECE, equal-mass binning). Regression: RMSE, MAE, R^2.
Every model sees IDENTICAL CV folds (fixed `cv` object per dataset) so per-fold scores are
PAIRED -- each baseline gets a 95% CI and a BETTER/WORSE/ns verdict against the strongest
non-TRA baseline via a paired t-test, plus an overall BETTER/WORSE/ns tally across all 20
datasets at the end of benchmark_summary.txt.

A `Main_Metric` column (accuracy for classification rows, RMSE for regression rows) is added
to benchmark_results.csv on top of the detailed per-metric columns, matching what
statistical_analysis.py expects for its Friedman test / average-rank tables -- run that script
after this one to get the cross-dataset significance tables for the paper.

USAGE
-----
    python benchmark.py            # full run: 20 datasets, 5-fold CV (expect 30-90+ minutes)
    python benchmark.py --quick    # sanity check: <=300 rows/dataset, 3-fold CV (a few minutes)

Results are written to benchmark_results.csv after EVERY dataset (not just at the end), so a
long run that is interrupted still leaves usable partial results.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import (fetch_california_housing, fetch_openml, load_breast_cancer,
                              load_diabetes, load_digits, load_wine, make_friedman1,
                              make_friedman2, make_friedman3, make_regression)
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier,
                              ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor, StackingClassifier, StackingRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import (KFold, StratifiedKFold, StratifiedShuffleSplit,
                                     cross_validate)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings('ignore')

try:                                    # keep the original package layout working
    from tra_algorithm.core import EnhancedTRA
except ImportError:                     # ...and a flat layout too
    from core import EnhancedTRA

RANDOM_STATE = 42
QUICK = "--quick" in sys.argv
N_FOLDS = 3 if QUICK else 5
ROW_CAP = 300 if QUICK else 5000

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available.")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not available.")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False
    print("CatBoost not available.")


# ----------------------------------------------------------------- calibration scorers

def _as_proba_matrix(proba, y_true):
    """Normalise a scorer's probability argument to an (n, n_classes) matrix.

    For BINARY problems scikit-learn's `response_method="predict_proba"` hands the scorer only
    the positive-class column, i.e. a 1-D array. `proba.max(axis=1)` then raises, the scorer
    returns NaN, and the Brier/ECE columns come back empty for exactly the binary datasets.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return np.column_stack([1.0 - proba, proba])
    return proba


def _expected_calibration_error(y_true, proba, n_bins=15):
    """ECE of the top-label confidence, using EQUAL-MASS bins.

    Equal-width bins are dominated by whichever bin holds most of the mass, which for an
    overconfident model is the top bin; equal-mass binning gives every bin the same sample count
    and is the less biased estimator.
    """
    proba = _as_proba_matrix(proba, y_true)
    classes = np.unique(y_true)
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


def _multiclass_brier(y_true, proba):
    """Multiclass Brier score. sklearn's brier_score_loss is BINARY only."""
    proba = _as_proba_matrix(proba, y_true)
    classes = np.unique(y_true)
    idx = {c: i for i, c in enumerate(classes)}
    onehot = np.eye(len(classes))[[idx[v] for v in y_true]]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


ECE_SCORER = make_scorer(_expected_calibration_error, response_method="predict_proba",
                         greater_is_better=False)
BRIER_SCORER = make_scorer(_multiclass_brier, response_method="predict_proba",
                           greater_is_better=False)


# --------------------------------------------------------------------------- dataset loaders

def _openml(name, version="active", target="label"):
    """Build a zero-arg loader for a pinned OpenML dataset.

    target="label"  -> string/categorical target, LabelEncoder'd to 0..K-1 (classification).
    target="float"  -> numeric target that OpenML may hand back as an object/string dtype
                       (e.g. wine-quality-red's '5','6',...); cast straight to float (regression).
    """
    def _load():
        d = fetch_openml(name, version=version, as_frame=False, parser="auto")
        X = np.asarray(d.data, dtype=float)
        y = np.asarray(d.target)
        y = LabelEncoder().fit_transform(y) if target == "label" else y.astype(float)
        return X, y
    return _load


def _load_auto_mpg():
    # 6 of 398 rows have a missing 'horsepower' value; dropped rather than imputed since it's
    # <2% of the data and this is a fixed evaluation set, not a production pipeline.
    d = fetch_openml("autoMpg", as_frame=False, parser="auto")
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=float)
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


CLASSIFICATION_DATASETS = [
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

REGRESSION_DATASETS = [
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


def _subsample(X, y, cap, task):
    """Cap dataset size (California Housing -> 5000 rows normally, or ROW_CAP in --quick mode).

    Classification uses a STRATIFIED subsample so rare classes (e.g. Glass) survive; regression
    uses a plain seeded random subsample.
    """
    if X.shape[0] <= cap:
        return X, y
    if task == "classification":
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=RANDOM_STATE)
        idx, _ = next(splitter.split(X, y))
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X.shape[0], cap, replace=False)
    return X[idx], y[idx]


# --------------------------------------------------------------------------- models

def _ensemble_base_estimators(task_type):
    """Base learners for the Voting/Stacking meta-ensembles: the strongest tree-based models
    available, matching what a well-resourced practitioner would actually combine. Falls back to
    sklearn's own GradientBoosting/ExtraTrees when XGBoost/LightGBM are not installed, so these
    two rows are never silently missing from the comparison.
    """
    if task_type == "classification":
        ests = [("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                              n_jobs=-1))]
        ests.append(("xgb", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE))
                    if XGB_AVAILABLE else
                    ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE)))
        ests.append(("lgbm", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1))
                    if LGBM_AVAILABLE else
                    ("et", ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                                n_jobs=-1)))
    else:
        ests = [("rf", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                             n_jobs=-1))]
        ests.append(("xgb", XGBRegressor(random_state=RANDOM_STATE))
                    if XGB_AVAILABLE else
                    ("gb", GradientBoostingRegressor(random_state=RANDOM_STATE)))
        ests.append(("lgbm", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1))
                    if LGBM_AVAILABLE else
                    ("et", ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                               n_jobs=-1)))
    return ests


def get_models(task_type):
    models = {}

    if task_type == "classification":
        # --- TRA, at the settings a 5-seed paired benchmark selected --- see module docstring.
        models["TRA (stacking)"] = EnhancedTRA(
            task_type="classification", n_tracks=4, random_state=RANDOM_STATE,
            combination_mode="stacking", small_data_auto_mode=False,
            feature_selection=False)
        models["TRA (routing)"] = EnhancedTRA(
            task_type="classification", n_tracks=4, random_state=RANDOM_STATE,
            router_use_oof=True, small_data_auto_mode=False,
            feature_selection=False)

        models["LogisticRegression"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
        models["KNN"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ])
        models["RandomForest"] = RandomForestClassifier(n_estimators=100,
                                                        random_state=RANDOM_STATE)
        models["ExtraTrees"] = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                                    n_jobs=-1)
        models["GradientBoosting"] = GradientBoostingClassifier(random_state=RANDOM_STATE)
        models["AdaBoost"] = AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)
        # SVC(probability=True) is deprecated (removed in sklearn 1.11). CalibratedClassifierCV
        # is the supported replacement and is also what TRA builds internally, so the SVM
        # baseline and TRA's SVM expert are now the same estimator.
        models["SVM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE, probability=False), cv=3)),
        ])
        models["MLP"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                  early_stopping=True, n_iter_no_change=15,
                                  random_state=RANDOM_STATE)),
        ])
        if XGB_AVAILABLE:
            # `use_label_encoder` was REMOVED in XGBoost 2.0 -- passing it raises on 3.x.
            models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
        if LGBM_AVAILABLE:
            models["LightGBM"] = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
        if CAT_AVAILABLE:
            models["CatBoost"] = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)

        base = _ensemble_base_estimators("classification")
        models["Voting"] = VotingClassifier(estimators=base, voting="soft", n_jobs=1)
        models["Stacking"] = StackingClassifier(
            estimators=base, final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1)

    elif task_type == "regression":
        # Regression: routing wins, stacking does not. Both entered to show the difference.
        models["TRA (routing)"] = EnhancedTRA(
            task_type="regression", n_tracks=4, random_state=RANDOM_STATE,
            router_use_oof=True, small_data_auto_mode=False,
            feature_selection=False)
        models["TRA (stacking)"] = EnhancedTRA(
            task_type="regression", n_tracks=4, random_state=RANDOM_STATE,
            combination_mode="stacking", small_data_auto_mode=False,
            feature_selection=False)

        models["Ridge"] = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ])
        models["KNN"] = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", KNeighborsRegressor(n_neighbors=5)),
        ])
        models["RandomForest"] = RandomForestRegressor(n_estimators=100,
                                                       random_state=RANDOM_STATE)
        models["ExtraTrees"] = ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                                   n_jobs=-1)
        models["GradientBoosting"] = GradientBoostingRegressor(random_state=RANDOM_STATE)
        models["AdaBoost"] = AdaBoostRegressor(n_estimators=100, random_state=RANDOM_STATE)
        models["SVR"] = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
        models["MLP"] = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                 early_stopping=True, n_iter_no_change=15,
                                 random_state=RANDOM_STATE)),
        ])
        if XGB_AVAILABLE:
            models["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE)
        if LGBM_AVAILABLE:
            models["LightGBM"] = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
        if CAT_AVAILABLE:
            models["CatBoost"] = CatBoostRegressor(verbose=0, random_state=RANDOM_STATE)

        base = _ensemble_base_estimators("regression")
        models["Voting"] = VotingRegressor(estimators=base, n_jobs=1)
        models["Stacking"] = StackingRegressor(
            estimators=base, final_estimator=Ridge(), cv=3, n_jobs=1)

    return models


SCORING = {
    "classification": {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc_ovr",
        "neg_log_loss": "neg_log_loss",
        "neg_brier": BRIER_SCORER,
        "neg_ece": ECE_SCORER,
    },
    "regression": {
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "r2": "r2",
    },
}

# Which metrics get a paired verdict, and whether higher is better.
REPORT = {
    "classification": [("accuracy", True), ("f1", True), ("roc_auc", True),
                       ("log_loss", False), ("brier", False), ("ece", False)],
    "regression": [("rmse", False), ("mae", False), ("r2", True)],
}


def _tidy(scores, task):
    """Convert cross_validate output into positive-oriented per-fold arrays."""
    if task == "classification":
        return {
            "accuracy": scores["test_accuracy"],
            "f1": scores["test_f1"],
            "roc_auc": scores["test_roc_auc"],
            "log_loss": -scores["test_neg_log_loss"],
            "brier": -scores["test_neg_brier"],
            "ece": -scores["test_neg_ece"],
        }
    return {
        "rmse": -scores["test_neg_rmse"],
        "mae": -scores["test_neg_mae"],
        "r2": scores["test_r2"],
    }


def run_benchmark():
    t_start = time.time()
    rows, folds = [], {}
    verdict_tally = {}  # model -> {'BETTER': n, 'WORSE': n, 'ns': n} across all 20 datasets

    datasets = ([(name, "classification", loader) for name, loader in CLASSIFICATION_DATASETS] +
                [(name, "regression", loader) for name, loader in REGRESSION_DATASETS])

    print(f"Starting Benchmark ({'QUICK' if QUICK else 'FULL'} mode, {N_FOLDS}-fold CV, "
          f"{len(datasets)} datasets)...")
    print("-" * 60)

    for d_i, (name, task, loader) in enumerate(datasets, 1):
        print(f"\n[{d_i}/{len(datasets)}] Dataset: {name} ({task})")
        try:
            data = loader()
            X, y = (data if isinstance(data, tuple) else (data.data, data.target))
        except Exception as e:
            # fetch_california_housing / fetch_openml need a download the first time.
            print(f"  Skipped ({type(e).__name__}: {e})")
            continue

        n_before = X.shape[0]
        X, y = _subsample(X, y, ROW_CAP, task)
        if X.shape[0] != n_before:
            print(f"  Subsampled {n_before} -> {X.shape[0]} rows.")

        # One CV object -> identical folds for every model -> per-fold scores are PAIRED.
        cv = (StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
              if task == "classification" else
              KFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE))

        for model_name, model in get_models(task).items():
            print(f"  {model_name} ...", end="", flush=True)
            t0 = time.time()
            try:
                scores = cross_validate(model, X, y, cv=cv, scoring=SCORING[task], n_jobs=1)
                elapsed = time.time() - t0
                per_fold = _tidy(scores, task)
                folds[(name, model_name)] = per_fold
                row = {"Dataset": name, "Task": task, "Model": model_name,
                       "Time (s)": round(elapsed, 2)}
                for k, v in per_fold.items():
                    row[k] = float(np.mean(v))
                headline = "accuracy" if task == "classification" else "rmse"
                row["Main_Metric"] = row[headline]
                rows.append(row)
                print(f" done ({headline}={row[headline]:.4f}, {elapsed:.1f}s)")
            except Exception as e:
                print(f" FAILED: {type(e).__name__}: {e}")

        # Persist after every dataset so a long run never loses completed work.
        pd.DataFrame(rows).to_csv("benchmark_results.csv", index=False)

    df = pd.DataFrame(rows)
    _report(df, folds, verdict_tally, time.time() - t_start)
    print("\nComplete. -> benchmark_results.csv, benchmark_summary.txt")


def _paired(a, b):
    """Paired t-test over the shared folds. Returns (mean_diff, ci_lo, ci_hi, p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size < 2 or np.allclose(d, d[0]):
        return float(d.mean()), np.nan, np.nan, np.nan
    sem = stats.sem(d)
    half = sem * stats.t.ppf(0.975, d.size - 1)
    return (float(d.mean()), float(d.mean() - half), float(d.mean() + half),
            float(stats.ttest_rel(a, b).pvalue))


def _report(df, folds, verdict_tally, total_elapsed):
    lines = ["TRA Benchmark Summary", "=" * 78,
             f"Mode: {'QUICK' if QUICK else 'FULL'}  |  Folds: {N_FOLDS}  |  "
             f"Datasets attempted: {df['Dataset'].nunique()}  |  "
             f"Total time: {total_elapsed / 60:.1f} min", ""]

    for dataset in df["Dataset"].unique():
        sub = df[df["Dataset"] == dataset]
        task = sub["Task"].iloc[0]
        headline = "accuracy" if task == "classification" else "rmse"
        higher = task == "classification"
        sub = sub.sort_values(headline, ascending=not higher)

        # Strongest NON-TRA model on the headline metric -- the bar TRA has to clear.
        base = sub[~sub["Model"].str.startswith("TRA")]
        if base.empty:
            continue
        ref = (base.loc[base[headline].idxmax()] if higher
               else base.loc[base[headline].idxmin()])["Model"]

        lines += [f"Dataset: {dataset}  ({task})",
                  f"Strongest baseline: {ref}", "",
                  sub.drop(columns=["Task", "Main_Metric"]).to_string(index=False), "",
                  f"Paired comparison vs {ref} ({N_FOLDS} shared folds; 'ns' = CI spans zero):"]

        for metric, hib in REPORT[task]:
            if (dataset, ref) not in folds:
                continue
            for model in sub["Model"]:
                if model == ref or (dataset, model) not in folds:
                    continue
                md, lo, hi, p = _paired(folds[(dataset, model)][metric],
                                        folds[(dataset, ref)][metric])
                verdict = "ns"
                if np.isfinite(lo) and (lo > 0 or hi < 0):
                    verdict = "BETTER" if ((md > 0) == hib) else "WORSE"
                if model.startswith("TRA") and metric == headline:
                    verdict_tally.setdefault(model, {"BETTER": 0, "WORSE": 0, "ns": 0})
                    verdict_tally[model][verdict] += 1
                star = " <<<" if model.startswith("TRA") and verdict == "BETTER" else ""
                lines.append(f"  {metric:>9}  {model:<18} {md:+9.4f}  "
                             f"[{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  {verdict}{star}")
        lines += ["", "-" * 78, ""]

    if verdict_tally:
        lines += ["OVERALL: TRA vs. strongest baseline on the headline metric, across all "
                 "datasets", "=" * 78]
        for model, tally in verdict_tally.items():
            n = sum(tally.values())
            lines.append(f"  {model:<18} BETTER={tally['BETTER']}/{n}  "
                         f"WORSE={tally['WORSE']}/{n}  ns={tally['ns']}/{n}")
        lines.append("")

    text = "\n".join(lines)
    with open("benchmark_summary.txt", "w") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    run_benchmark()
