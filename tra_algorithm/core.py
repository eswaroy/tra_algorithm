"""
Enhanced Track/Rail Algorithm (TRA) with Mixture-of-Experts and Switch Transformer Routing.

All 10 major improvements fully integrated:
1. Stronger Router (XGBoost, CatBoost, MLP, LightGBM)
2. Heterogeneous Expert Tracks
3. Increase Number of Tracks (5-8+)
4. Load Balancing Loss
5. Top-K Routing
6. Expert Capacity Control
7. Router Meta-Features
8. Temperature-Scaled Soft Routing
9. Dynamic Track Creation
10. Track Specialization via Clustering
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.validation import check_X_y as _sk_check_X_y, check_array as _sk_check_array, check_is_fitted, has_fit_parameter

# PHASE7-FIX: sklearn renamed check_X_y/check_array's "force_all_finite" kwarg to
# "ensure_all_finite" in newer releases (this environment has 1.8.0, which only accepts the
# new name) but older sklearn only accepts the old name — detect which one is supported once at
# import time rather than try/except-ing on every single validation call.
import inspect as _inspect
_FINITE_KWARG = 'ensure_all_finite' if 'ensure_all_finite' in _inspect.signature(_sk_check_array).parameters else 'force_all_finite'


def check_X_y(X, y, **kwargs):
    """Thin wrapper allowing NaN through to the pipeline's SimpleImputer (spec item 41:
    'handle missing values') across sklearn versions that disagree on the kwarg name."""
    kwargs.setdefault(_FINITE_KWARG, 'allow-nan')
    return _sk_check_X_y(X, y, **kwargs)


def check_array(X, **kwargs):
    """See check_X_y — same version-compatibility shim, for the predict()-side validation calls."""
    kwargs.setdefault(_FINITE_KWARG, 'allow-nan')
    return _sk_check_array(X, **kwargs)


from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import warnings
import joblib
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class SignalExtractor:
    """
    Signal-Guided Routing Enhancement: Structural Signal Extraction Layer.
    
    Extracts 5 structural signals about input samples to guide expert routing:
    1. Expert Disagreement - std(expert_predictions)
    2. Prediction Entropy - entropy(router_probabilities)
    3. Feature Density Score - distance to k-NN in feature space
    4. Cluster Distance - distance to nearest KMeans centroid
    5. Outlier Score - IsolationForest anomaly detection
    
    These signals represent structural difficulty and data region, making routing
    aware of the data geometry and expert consensus.
    """
    
    def __init__(self, n_neighbors: int = 5, contamination: float = 0.1, random_state: Optional[int] = None):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.random_state = random_state
        self.kmeans_ = None
        self.isolation_forest_ = None
        self.knn_tree_ = None
        self.X_train_ = None
        self.fitted_ = False

    def fit(self, X: np.ndarray, kmeans: Optional[KMeans] = None):
        """Fit signal extraction components."""
        from sklearn.neighbors import NearestNeighbors
        try:
            from sklearn.ensemble import IsolationForest
            HAS_ISOLATION = True
        except:
            HAS_ISOLATION = False

        self.X_train_ = X

        # Fit KNN for density estimation
        self.knn_tree_ = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X)-1), algorithm='auto')
        self.knn_tree_.fit(X)

        # PHASE7 FIX (Signal 4 was a dead constant): a caller-supplied KMeans (from
        # cluster_experts=True track specialization) is reused when available -- no reason to
        # fit a second one. But cluster_experts defaults to False and is off in every default
        # configuration, which previously left self.kmeans_ permanently None and Signal 4
        # ("Cluster Distance") permanently a constant 0.5 for every sample, in every run that
        # did not explicitly enable cluster_experts. Cluster distance is one of the algorithm's
        # five namesake structural signals and should not silently be a no-op by default, so a
        # small dedicated KMeans is fit here whenever one wasn't already provided. Cheap
        # relative to the IsolationForest/kNN fits already done in this method.
        if kmeans is not None:
            self.kmeans_ = kmeans
        else:
            try:
                n_clusters = max(2, min(8, len(X) // 20))
                if len(X) >= n_clusters:
                    self.kmeans_ = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
                    self.kmeans_.fit(X)
            except Exception:
                self.kmeans_ = None

        # Fit IsolationForest if available
        if HAS_ISOLATION:
            try:
                self.isolation_forest_ = IsolationForest(contamination=self.contamination, random_state=self.random_state)
                self.isolation_forest_.fit(X)
            except:
                self.isolation_forest_ = None
        
        self.fitted_ = True
        return self
    
    def extract_signals(self, X: np.ndarray, track_predictions: Optional[np.ndarray] = None,
                       ensemble_proba: Optional[np.ndarray] = None,
                       router_probs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract structural signals for routing guidance.

        Returns array of shape (n_samples, 5) with signals:
        [disagreement, entropy, density, cluster_distance, outlier_score]

        PHASE7 FIX (Signal 2 was a dead constant): `ensemble_proba` -- the (n_samples,
        n_classes) MEAN probability across expert tracks (classification only) -- is what
        Signal 2 (entropy) is computed from. Every call site in this module always passed
        `router_probs=None` (the router's own probabilities are circular to require here: the
        router doesn't exist yet when its own training meta-features are being built), so
        entropy was silently 0.0 for every sample in every configuration. `ensemble_proba` is
        available everywhere entropy is needed (it's just the already-computed mean of the
        per-track OOF/production probabilities) and captures the same underlying idea --
        "how much genuine disagreement/uncertainty is there in the expert pool's consensus" --
        without depending on a router. `router_probs` is kept as a deprecated alias so any
        external caller still passing it keeps working.
        """
        if ensemble_proba is None:
            ensemble_proba = router_probs
        if not self.fitted_:
            raise ValueError("SignalExtractor not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        signals = np.zeros((n_samples, 5))
        
        # Signal 1: Expert Disagreement (std of expert predictions)
        if track_predictions is not None and track_predictions.shape[1] > 1:
            signals[:, 0] = np.std(track_predictions, axis=1)
        else:
            signals[:, 0] = 0.0
        
        # Signal 2: Prediction Entropy (entropy of the ensemble's mean class-probability
        # distribution -- see PHASE7 FIX note in extract_signals' docstring above).
        if ensemble_proba is not None and np.asarray(ensemble_proba).ndim == 2 and ensemble_proba.shape[1] > 1:
            probs = np.clip(ensemble_proba, 1e-10, 1.0)
            probs = probs / probs.sum(axis=1, keepdims=True)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            # Normalized to [0, 1] by the maximum possible entropy (uniform over n_classes), so
            # this signal is on the same scale as the other four regardless of class count.
            max_entropy = np.log(probs.shape[1])
            signals[:, 1] = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            signals[:, 1] = 0.0
        
        # Signal 3: Feature Density Score (inverse of distance to k-NN)
        try:
            distances, _ = self.knn_tree_.kneighbors(X)
            # Use mean distance to neighbors (excluding self at index 0)
            mean_knn_distance = np.mean(distances[:, 1:], axis=1)
            # Invert: closer neighbors = higher density
            max_distance = np.max(mean_knn_distance) + 1e-10
            signals[:, 2] = 1.0 - (mean_knn_distance / max_distance)
        except:
            signals[:, 2] = 0.5
        
        # Signal 4: Cluster Distance (distance to KMeans centroid)
        if self.kmeans_ is not None:
            try:
                distances = np.min(np.sqrt(((X - self.kmeans_.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2)), axis=0)
                max_distance = np.max(distances) + 1e-10
                signals[:, 3] = distances / max_distance
            except:
                signals[:, 3] = 0.5
        else:
            signals[:, 3] = 0.5
        
        # Signal 5: Outlier Score (IsolationForest anomaly score)
        if self.isolation_forest_ is not None:
            try:
                outlier_scores = -self.isolation_forest_.score_samples(X)
                signals[:, 4] = (outlier_scores - outlier_scores.min()) / (outlier_scores.max() - outlier_scores.min() + 1e-10)
            except:
                signals[:, 4] = 0.5
        else:
            signals[:, 4] = 0.5
        
        return signals

# PHASE2 cleanup (spec item 32): a SECOND, shadowing definition of PageHinkleyDetector
# was removed from this location. Two classes of the same name were defined 36 lines
# apart with different signatures (threshold=50.0 vs threshold=0.02 plus an `alpha`
# decay term); Python bound the name to the later one, so the first was unreachable
# dead code that nonetheless read as the live implementation. The surviving definition
# is the one below, which is the one the model has always actually used.

class PageHinkleyDetector:
    """PHASE4 (spec item 24): lightweight, dependency-free Page-Hinkley concept-drift test.

    Monitors a scalar stream statistic (here: per-partial_fit-chunk mean loss) and flags
    drift once the cumulative deviation of observations from the running mean exceeds
    `threshold`, using `delta` as a slack term (ignore small fluctuations) and `alpha` as a
    forgetting factor. This is the standard, well-established Page-Hinkley test — not a novel
    invention — chosen specifically because it needs no extra dependency (spec item 24: "do
    not introduce a huge dependency solely for drift detection").
    """
    def __init__(self, delta: float = 0.005, threshold: float = 0.02, alpha: float = 0.9999):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mean_ = 0.0
        self.n_ = 0
        self.sum_ = 0.0
        self.min_sum_ = 0.0

    def update(self, value: float) -> bool:
        """Feed one new observation. Returns True (and resets internal state) if drift is flagged."""
        self.n_ += 1
        self.mean_ = self.mean_ + (value - self.mean_) / self.n_
        self.sum_ = self.alpha * self.sum_ + (value - self.mean_ - self.delta)
        self.min_sum_ = min(self.min_sum_, self.sum_)
        drift = (self.sum_ - self.min_sum_) > self.threshold
        if drift:
            self.reset()
        return drift


class _LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """PHASE4 fix: adapts a classifier that requires contiguous 0..k-1 integer labels.

    XGBoost's sklearn API rejects any label space that is not exactly [0, 1, ..., k-1] with
    "Invalid classes inferred from unique values of `y`". So a TRA model with an XGBoost track
    raised on string labels ("setosa") and on non-contiguous integer labels ({10, 42, 77}) —
    both of which spec item 15 explicitly requires to work, and both of which TRA handles fine
    everywhere else.

    This was invisible in any environment without XGBoost installed, because core.py degrades
    gracefully and simply builds a different expert pool. It reproduces immediately once
    XGBoost is present, which is the configuration that actually ships.

    The wrapper encodes y to indices at fit time and decodes on the way out, exposing
    `classes_` in the ORIGINAL label space so `_align_proba` and every other consumer keep
    working unchanged. Probability column order is preserved: encoded class i corresponds to
    classes_[i] by construction.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        lookup = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([lookup[v] for v in y], dtype=int)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_encoded, **kwargs)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.estimator_.predict(X), dtype=int)]

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def __sklearn_tags__(self):
        # Defer to the wrapped estimator where sklearn asks about capabilities.
        return self.base_estimator.__sklearn_tags__()


class Track:
    """Enhanced Track with capacity limits and performance metrics (IMPROVEMENTS 3, 6)."""
    
    def __init__(self, name: str, classifier=None, feature_indices: Optional[np.ndarray] = None,
                 expert_capacity: Optional[float] = None):
        self.name = name
        self.classifier = classifier
        self.feature_indices = feature_indices
        self.performance_score = 0.5  # PHASE3: repurposed to hold the OOF leave-one-out
        # marginal-contribution score from _compute_expert_diagnostics once fit() computes it
        # (was previously set once at construction and never read anywhere — dead state).
        # Positive = removing this track increases full-ensemble OOF loss (valuable); the
        # default 0.5 is a neutral placeholder used only if diagnostics were never computed.
        self.usage_count = 0
        self.capacity_violations = 0
        self.last_used = time.time()
        self.prediction_times = deque(maxlen=50)
        self.expert_capacity = expert_capacity
        
    def predict(self, X: np.ndarray):
        """Make predictions using this track's classifier with timing."""
        if self.classifier is None:
            raise ValueError(f"No classifier available for track {self.name}")
        
        start_time = time.time()
        
        X_subset = X[:, self.feature_indices] if self.feature_indices is not None else X
        result = self.classifier.predict(X_subset)
        
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        return result
    
    def predict_proba(self, X: np.ndarray):
        """Make probability predictions using this track's classifier with timing."""
        if self.classifier is None:
            raise ValueError(f"No classifier available for track {self.name}")
        
        start_time = time.time()
        
        X_subset = X[:, self.feature_indices] if self.feature_indices is not None else X
        result = self.classifier.predict_proba(X_subset)
        
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        return result
    
    def get_average_prediction_time(self):
        """Get average prediction time for this track."""
        return np.mean(self.prediction_times) if self.prediction_times else 0.0
    
    def is_underused(self, min_usage_threshold: int = 5, time_threshold: float = 300.0):
        """Check if track is underused and candidate for pruning."""
        current_time = time.time()
        return (self.usage_count < min_usage_threshold and 
                current_time - self.last_used > time_threshold)

class EnhancedTRA(BaseEstimator, ClassifierMixin, RegressorMixin):
    """
    Signal-Guided Expert Ensemble: Enhanced Track/Rail Algorithm (TRA).
    
    A novel Mixture-of-Experts architecture that combines the switch transformer 
    routing of Google's MoE systems with signal-guided gating. Unlike standard MoE 
    which routes solely on input features, TRA extracts structural signals about 
    the input (disagreement, entropy, density, outlier scores) to guide expert 
    routing for improved specialization.
    
    Core Architecture:
    Input → Signal Extraction → Structure-Aware Router → Expert Tracks → 
    Residual Correction → Final Prediction
    
    11 Integrated Improvements:
    1. Stronger Router Model: XGBoost, CatBoost, MLP, LightGBM
    2. Heterogeneous Expert Tracks: Diverse model types per track (RF, LightGBM, XGBoost, SVM, MLP)
    3. Increase Number of Tracks: 5-8+ expert tracks
    4. Load Balancing Loss: Prevent expert collapse
    5. Top-K Routing: Route to multiple experts with weighted averaging
    6. Expert Capacity Control: Limit samples per expert
    7. Router Meta-Features: Track disagreement signals
    8. Temperature-Scaled Soft Routing: Smooth decision boundaries
    9. Dynamic Track Creation: Spawn specialists for uncertain regions
    10. Track Specialization via Clustering: KMeans-based region-based experts
    11. SIGNAL-GUIDED ROUTING: Structural signal extraction layer
        • Expert disagreement (std of predictions)
        • Prediction entropy (entropy of router probabilities)
        • Feature density score (distance to k-nearest neighbors)
        • Cluster distance (distance to KMeans centroid)
        • Outlier score (IsolationForest anomaly detection)
    """
    
    # PHASE6 item 29: bump this whenever a save_model()/load_model() schema-relevant change is
    # made (new fitted-state attributes that predict()/predict_proba() depend on). Used by
    # load_model() to warn on version mismatch, NOT to hard-block loading -- old saves should
    # keep working where the underlying pickled object still has everything it needs.
    # PHASE7: bumped -- stacking_use_signals=True (new default) changes the stacking
    # meta-learner's input width (adds 5 structural-signal columns), so a meta_learner_ fitted
    # under 6.0-phase6 or earlier expects a narrower input than the current
    # _stacking_predict_input() produces. Loading an old save still works (the pickled
    # meta_learner_ and stacking_use_signals flag are restored together), but a version mismatch
    # here is the expected signal to re-check that assumption.
    _ARCHITECTURE_VERSION = "6.2"  # IMPROVEMENT: bumped for the 1.0.6 correctness/robustness fixes.
    
    def __init__(self,
                 task_type: str = "classification",
                 n_tracks: int = 5,
                 # IMPROVEMENT 1: Stronger Router
                 router_type: str = "xgboost",
                 # IMPROVEMENT 2: Heterogeneous Experts
                 track_models: Optional[List] = None,
                 # IMPROVEMENT 3: More Tracks
                 max_tracks: int = 8,
                 # IMPROVEMENT 4: Load Balancing
                 load_balance_strength: float = 0.01,
                 # IMPROVEMENT 5: Top-K Routing
                 top_k: int = 1,
                 # IMPROVEMENT 6: Expert Capacity
                 expert_capacity: Optional[float] = None,
                 # IMPROVEMENT 7: Meta-Features
                 use_meta_features: bool = True,
                 # IMPROVEMENT 8: Temperature Scaling
                 routing_temperature: float = 1.0,
                 # IMPROVEMENT 9 / PHASE4 (spec section 11): Dynamic Track Creation, redesigned
                 # into a safe "buffer real-labeled uncertain samples, then validate before
                 # deploying" specialist pipeline. confidence_spawn_threshold keeps its original
                 # meaning (trigger sensitivity: what fraction of a batch must be low-confidence
                 # to consider spawning at all). Everything else is new and OFF by default:
                 #   - enable_dynamic_spawning: master switch. Was previously implicitly "on"
                 #     whenever confidence_spawn_threshold > 0 (its old default); that silently
                 #     mutated model architecture during ordinary predict() calls, which violates
                 #     "don't mutate architecture during predict() unless an explicit
                 #     online-adaptation mode is enabled". This is that explicit opt-in.
                 #   - self_training: separate, clearly-labeled-risky opt-in for spawning from
                 #     the model's OWN predictions when no true labels are available (only
                 #     relevant to the predict()-time path — partial_fit always has real labels
                 #     and never needs this). Off by default per "do not use self-generated
                 #     pseudo-labels to create new experts by default."
                 enable_dynamic_spawning: bool = False,
                 self_training: bool = False,
                 confidence_spawn_threshold: float = 0.3,
                 max_dynamic_tracks: int = 3,
                 min_spawn_samples: int = 30,
                 min_spawn_gain: float = 0.02,
                 spawn_cooldown: int = 5,
                 specialist_validation_fraction: float = 0.3,
                 # PHASE4 (spec item 24): lightweight, dependency-free Page-Hinkley drift
                 # detector over partial_fit chunk loss. drift_threshold is its sensitivity
                 # (smaller = more sensitive) and also relaxes min_spawn_gain for one chunk
                 # when drift fires, per spec item 24 ("allow specialist creation" on drift).
                 drift_threshold: float = 0.02,
                 # IMPROVEMENT 10: Track Specialization via Clustering
                 cluster_experts: bool = False,
                 # PHASE1 item 7: genuine regression uncertainty. When enabled (regression only),
                 # a held-out calibration slice is carved off the training data BEFORE any track/
                 # router/stacking training and used for split-conformal prediction intervals
                 # (predict_interval()). Falls back to an explicitly-uncalibrated ensemble-
                 # disagreement interval when there isn't enough data to calibrate safely.
                 enable_conformal: bool = True,
                 conformal_calibration_fraction: float = 0.15,
                 conformal_min_samples: int = 40,
                 # PHASE2 item 3: loss-aware routing (opt-in — see _apply_loss_aware_blend /
                 # _fit_loss_aware_router). Off by default so existing users' routing behavior
                 # is completely unchanged unless they explicitly enable it.
                 router_loss_aware: bool = False,
                 loss_aware_blend: float = 0.5,
                 loss_aware_temperature: float = 1.0,
                 # PHASE2 item 5: leak-free difficulty model — cheap (fit on 5 structural
                 # signals only), on by default whenever router_use_oof-style OOF data is being
                 # computed anyway for other reasons; otherwise it computes its own small OOF
                 # pass. Purely diagnostic/advisory in Phase 2 (see predict_difficulty()); not
                 # yet wired into routing decisions (that lands with hierarchical routing/
                 # adaptive top-k in a later phase).
                 enable_difficulty_model: bool = True,
                 # Original parameters (for backward compatibility)
                 signal_threshold: float = 0.1,
                 random_state: Optional[int] = None,
                 # PHASE7: default raised 50 -> 100. Every RandomForest/LightGBM/XGBoost/CatBoost
                 # expert track (and the residual correction track) used this as its tree count,
                 # which was HALF of what this project's own benchmark baselines use for the
                 # exact same model families (RandomForestClassifier(n_estimators=100),
                 # ExtraTrees(n_estimators=200), CatBoost's own library default of 1000
                 # iterations) -- an unforced capacity handicap relative to the models TRA is
                 # compared against, independent of any architectural difference. Combined with
                 # adaptive_n_estimators defaulting to False (PHASE5 already established that
                 # shrinking capacity on small data hurts more than it helps), there was no
                 # mechanism that ever raised capacity back toward baseline parity. 100 matches
                 # the RandomForest/XGBoost/LightGBM sklearn-API baseline default exactly; set
                 # explicitly to 50 to reproduce pre-PHASE7 capacity.
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 4,
                 # PHASE5: default flipped True -> False. The univariate SelectKBest filter this
                 # enables was costing 22 accuracy points on regime_clf (0.8113 -> 0.5887) and 67%
                 # more MAE on piecewise_reg (16.62 -> 27.77). The cause is structural, not a bad
                 # k: SelectKBest ranks features by GLOBAL univariate association, but in a
                 # mixture-of-experts problem feature relevance is REGIME-CONDITIONAL — a feature
                 # can be strongly predictive inside each regime while its per-regime effects
                 # cancel globally, scoring near zero. Univariate filtering is therefore hostile
                 # to the premise TRA is built on. Set True (ideally with a non-univariate
                 # feature_selection_method) if you have genuinely redundant columns.
                 feature_selection: bool = False,
                 # PHASE5 (spec item 19): how features are ranked when feature_selection=True.
                 #   "model_based"  RandomForest importances — captures interactions and
                 #                  regime-conditional relevance. Default.
                 #   "mutual_info"  mutual information — captures non-linear but still marginal
                 #                  dependence.
                 #   "univariate"   the original ANOVA F-test. Retained for reproducibility of
                 #                  earlier runs; not recommended for regime-structured data.
                 feature_selection_method: str = "model_based",
                 handle_imbalanced: bool = True,
                 max_workers: int = 4,
                 enable_track_pruning: bool = True,
                 # PHASE3 item 17: contribution-aware pruning floor. Replaces the previous
                 # hardcoded "never go below 2 tracks" behavior with an explicit, user-tunable
                 # parameter — default (2) preserves the exact prior behavior.
                 minimum_experts: int = 2,
                 enable_correction_track: bool = True,
                 # PHASE1: correction track is now trained/gated using leakage-free OOF ensemble
                 # predictions (see _add_correction_track). correction_gain_threshold is the
                 # minimum held-out improvement (accuracy on previously-OOF-wrong samples for
                 # classification; fractional MAE reduction for regression) required to keep the
                 # correction track at all; below this it is discarded rather than deployed.
                 # correction_confidence_threshold is the minimum predict_proba confidence the
                 # correction track must have before it is allowed to override a classification
                 # prediction at inference time.
                 correction_gain_threshold: float = 0.02,
                 correction_confidence_threshold: float = 0.7,
                 pruning_interval: int = 100,
                 abstention_threshold: float = 0.0,
                 abstention_class: Any = None,
                 # PHASE5 (spec item 26): abstain based on the validated expected-risk/difficulty
                 # model (Phase 2) instead of raw routing confidence. Off by default — with it
                 # off, abstention_threshold keeps its exact original meaning (compare against
                 # routing_confidences). Genuinely improves on the old behavior for regression
                 # specifically: routing_confidences reflects the router's confidence in WHICH
                 # track it picked, not whether that track's prediction is actually reliable —
                 # difficulty is trained against real OOF ensemble loss instead (see
                 # _fit_difficulty_model), so it means "abstain if expected risk > threshold"
                 # for classification AND regression uniformly, matching spec item 26's wording.
                 # Falls back to the original routing_confidences behavior (with a warning) if
                 # the difficulty model isn't available for this fit.
                 abstention_use_difficulty: bool = False,
                 # PHASE7: the difficulty model (enable_difficulty_model) was, until now, a
                 # pure diagnostic -- predict_difficulty() exposed it, but nothing in the actual
                 # fusion decision (_fuse, soft routing, stacking) ever consulted it. When True,
                 # per-sample predicted difficulty (already normalized to [0, 1] against the
                 # training-set difficulty distribution) is used to blend the routed/stacked
                 # fusion weights toward a flat, uniform average across tracks: an easy sample
                 # (difficulty near 0) is fused exactly as before; a sample the difficulty model
                 # expects the ensemble to struggle with (difficulty near 1) is hedged toward
                 # "trust no single track/meta-learner decision more than any other" rather than
                 # committing fully to whatever the router/meta-learner happened to prefer. Off
                 # by default (requires enable_difficulty_model=True to have any effect; adds a
                 # real per-predict-call cost to compute difficulty) since this is a new,
                 # unvalidated-in-this-project's-benchmark mechanism, not a default behavior
                 # change like the fixes above it.
                 difficulty_aware_fusion: bool = False,
                 # PHASE5 (spec item 21): ensemble-level (post-fusion) probability calibration
                 # with VALIDATED method selection, not a blindly-applied fixed transform. Off
                 # by default. "auto" fits {none, temperature, sigmoid, isotonic-if-enough-data}
                 # on a genuinely held-out calibration slice (carved off before any track/router/
                 # stacking training, same pattern as PHASE1's conformal calibration) and keeps
                 # whichever has the lowest held-out log loss — "none" is a literal candidate, so
                 # calibration can never make held-out log loss worse than doing nothing. Metrics
                 # for every candidate are recorded in self.calibration_report_ (get_calibration_
                 # report()) rather than assumed to have worked.
                 calibrate_output: bool = False,
                 calibration_method: str = "auto",
                 calibration_fraction: float = 0.15,
                 calibration_min_samples: int = 60,
                 routing_mode: str = "soft",
                 # FIX 2: was hardcoded n_jobs=1 on every RF/XGBoost/LightGBM/router model; now configurable, defaults to using all cores
                 n_jobs: int = -1,
                 # FIX 6: opt-out flag for scaling n_estimators down on small datasets
                 # PHASE5: default flipped True -> False. Scaling expert capacity down is a
                 # data-efficiency loss that the routing layer does not earn back: measured at
                 # 5 seeds, restoring full capacity alongside bagging_mode="full" closed the
                 # ENTIRE regression gap against a uniform average of the same four model
                 # families (MAE 17.20 -> 15.64 vs the control's 15.71) and half the
                 # classification gap (0.7748 -> 0.8068). Set True to restore the old scaling.
                 adaptive_n_estimators: bool = False,
                 # ACC-FIX 1: "bootstrap" preserves current per-track row bootstrap sampling exactly; "full" trains every track on the complete data
                 # PHASE5: default flipped "bootstrap" -> "full". Bootstrap gives each expert
                 # only ~63% of the rows to buy error diversity; across eight datasets that
                 # diversity never paid for the accuracy it cost ("full" was directionally
                 # better on 6/8, significantly better on 1, never significantly worse), and on
                 # the two regime datasets it accounts for roughly half the deficit against a
                 # matched uniform ensemble. Set "bootstrap" to restore the old behaviour.
                 bagging_mode: str = "full",
                 # ACC-FIX 2: None preserves the current random 0.7-0.9 keep-fraction; a float fixes it; 1.0 disables feature dropout entirely
                 feature_dropout_rate: Optional[float] = None,
                 # ACC-FIX 3: configurable router holdout fraction (was hardcoded 0.2)
                 router_holdout_fraction: float = 0.2,
                 # ACC-FIX 4: "routing" preserves current router-based hard/soft selection; "stacking" trains a meta-learner on OOF track outputs instead
                 # ACC2-FIX 1: "auto" is a new valid value alongside "routing"/"stacking" — default is
                 # unchanged ("routing"), so existing users see no behavior change unless they opt in.
                 combination_mode: str = "routing",
                 stacking_folds: int = 5,
                 # ACC2-FIX 1: lighter fold count (vs stacking_folds) used only for the internal
                 # routing-vs-stacking comparison when combination_mode="auto", to keep that
                 # comparison fast even when stacking_folds is set high for the real stacking fit.
                 auto_select_folds: int = 3,
                 # ACC2-FIX 2: when True, wrap each track's classifier in CalibratedClassifierCV
                 # (classification only) so predict_proba is calibrated before combination.
                 calibrate_tracks: bool = False,
                 # ACC2-FIX 3: when True (and combination_mode resolves to "stacking"), use
                 # LogisticRegressionCV / RidgeCV for the meta-learner instead of the fixed
                 # LogisticRegression / Ridge.
                 # PHASE7: default flipped False -> True. The stacking meta-learner is a small
                 # linear model with real regularization sensitivity (a fixed C=1.0/alpha=1.0 is
                 # an arbitrary choice, not a validated one); cross-validating it over a small
                 # {0.01, 0.1, 1, 10, 100} grid costs one extra cheap linear fit per candidate and
                 # can only match or beat the fixed-regularization fit on its own training data.
                 # Set False to reproduce the exact pre-PHASE7 stacking meta-learner.
                 meta_learner_cv: bool = True,
                 # PHASE7: when True (stacking mode only), the leak-free structural signals
                 # (expert disagreement, prediction entropy, feature density, cluster distance,
                 # outlier score) are concatenated onto the per-track OOF meta-feature matrix
                 # before fitting/using the stacking meta-learner. Before this, stacking -- the
                 # combination_mode this project's own benchmark recommends and uses for its best
                 # calibration numbers -- never consumed the structural signals at all (see
                 # _fit_stacking's prior docstring), meaning the "signal-guided" mechanism that
                 # names this algorithm contributed nothing to its best-performing configuration.
                 # Default True completes that design; set False to reproduce byte-identical
                 # pre-PHASE7 stacking behavior (e.g. to compare against saved older results).
                 stacking_use_signals: bool = True,
                 # ACC2-FIX 4: when True, incrementally upweight samples that previously-trained
                 # tracks got wrong when training each subsequent track (bootstrap/full bagging only).
                 diversity_reweighting: bool = False,
                 # ACC-FIX 5: when True (and combination_mode="routing"), train the router on k-fold OOF "best track" labels over the full training set instead of a single holdout
                 router_use_oof: bool = False,
                 # ACC-FIX 6: auto-apply data-efficient defaults (bagging_mode="full", combination_mode="stacking") on small datasets unless the user already overrode them
                 small_data_auto_mode: bool = True,
                 small_data_threshold: int = 2000,
                 # ACC-FIX 7: below this router confidence, blend the routed prediction toward an unweighted average of all tracks (routing mode only)
                 router_fallback_threshold: float = 0.3,
                 router_fallback_blend: float = 0.5,
                 track_early_stopping: bool = False,
                 diversity_reweighting_folds: int = 3,
                 intelligent_routing: bool = False,
                 # PHASE1-F1: how per-expert predicted losses are scaled before the utility
                 # softmax in _apply_loss_aware_blend. The previous implementation fed RAW loss
                 # units to the softmax, which made the blend a no-op for classification
                 # (losses on [0,1]) and a degenerate hard argmax for regression (losses of
                 # O(target scale)). "zscore" (default) standardises across experts per sample so
                 # loss_aware_temperature has a consistent, task-independent meaning; "range"
                 # min-max scales; "none" reproduces the pre-PHASE1 behaviour exactly.
                 loss_aware_normalization: str = "zscore",
                 # PHASE1-F2: controls per-expert feature subsetting, which is a DIVERSITY
                 # mechanism and was previously reachable only when feature_selection=True (a
                 # DIMENSIONALITY mechanism). "auto" applies it independently; "legacy" restores
                 # the old coupling; "off" disables it. feature_dropout_rate=1.0 still disables.
                 expert_feature_subsets: str = "legacy",
                 routing_regions: int = 4,
                 routing_bootstrap_ensemble_size: int = 3,
                 routing_memory_size: int = 64,
                 adaptive_top_k_max: int = 6):
        
        # Validate inputs
        if task_type not in ("classification", "regression"):
            raise ValueError(f"Invalid task_type: {task_type}")
        if routing_mode not in ("hard", "soft"):
            raise ValueError(f"Invalid routing_mode: {routing_mode}")
        if router_type not in ("xgboost", "catboost", "mlp", "lightgbm"):
            raise ValueError(f"Invalid router_type: {router_type}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1")
        # ACC-FIX 1/2/4: validate new mode/rate parameters
        if feature_selection_method not in ("univariate", "mutual_info", "model_based"):
            raise ValueError(f"Invalid feature_selection_method: {feature_selection_method}")
        if bagging_mode not in ("bootstrap", "full"):
            raise ValueError(f"Invalid bagging_mode: {bagging_mode}")
        # PHASE3-G2: "flat_average" added — a uniform average over all expert tracks, with no
        # router and no meta-learner. It is a real candidate rather than a degenerate case: on
        # data with no latent regime structure it is frequently the best available fusion, and
        # a selector that cannot choose it will keep paying for routing that does not help.
        if combination_mode not in ("routing", "stacking", "flat_average", "auto"):
            raise ValueError(f"Invalid combination_mode: {combination_mode}")
        if feature_dropout_rate is not None and not (0.0 < feature_dropout_rate <= 1.0):
            raise ValueError(f"feature_dropout_rate must be None or in (0, 1], got {feature_dropout_rate}")
        if loss_aware_normalization not in ("zscore", "range", "none"):
            raise ValueError(f"Invalid loss_aware_normalization: {loss_aware_normalization}")
        if expert_feature_subsets not in ("auto", "legacy", "off"):
            raise ValueError(f"Invalid expert_feature_subsets: {expert_feature_subsets}")
        if routing_regions < 2:
            raise ValueError(f"routing_regions must be >= 2, got {routing_regions}")
        if routing_bootstrap_ensemble_size < 1:
            raise ValueError(f"routing_bootstrap_ensemble_size must be >= 1, got {routing_bootstrap_ensemble_size}")
        if routing_memory_size < 1:
            raise ValueError(f"routing_memory_size must be >= 1, got {routing_memory_size}")
        if adaptive_top_k_max < 1:
            raise ValueError(f"adaptive_top_k_max must be >= 1, got {adaptive_top_k_max}")
        
        self.task_type = task_type
        self.n_tracks = max(2, n_tracks)
        self.max_tracks = max(self.n_tracks, max_tracks)
        self.router_type = router_type
        self.track_models = track_models
        self.load_balance_strength = load_balance_strength
        self.top_k = min(top_k, self.n_tracks)
        self.expert_capacity = expert_capacity
        self.use_meta_features = use_meta_features
        self.routing_temperature = max(0.1, routing_temperature)
        self.enable_dynamic_spawning = enable_dynamic_spawning
        self.self_training = self_training
        self.confidence_spawn_threshold = confidence_spawn_threshold
        self.min_spawn_samples = min_spawn_samples
        self.min_spawn_gain = min_spawn_gain
        self.spawn_cooldown = spawn_cooldown
        self.specialist_validation_fraction = specialist_validation_fraction
        self.drift_threshold = drift_threshold
        self.max_dynamic_tracks = max_dynamic_tracks
        self.cluster_experts = cluster_experts
        
        # Original parameters
        self.signal_threshold = signal_threshold
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_selection = feature_selection
        self.feature_selection_method = feature_selection_method  # PHASE5 (spec item 19)
        self.handle_imbalanced = handle_imbalanced
        self.max_workers = min(max_workers, 8)
        self.enable_track_pruning = enable_track_pruning
        self.minimum_experts = max(1, minimum_experts)
        self.enable_correction_track = enable_correction_track
        self.correction_gain_threshold = correction_gain_threshold
        self.correction_confidence_threshold = correction_confidence_threshold
        self.enable_conformal = enable_conformal
        self.conformal_calibration_fraction = conformal_calibration_fraction
        self.conformal_min_samples = conformal_min_samples
        self.router_loss_aware = router_loss_aware
        self.loss_aware_blend = loss_aware_blend
        self.loss_aware_temperature = loss_aware_temperature
        self.enable_difficulty_model = enable_difficulty_model
        self.pruning_interval = pruning_interval
        self.abstention_threshold = abstention_threshold
        self.abstention_class = abstention_class
        self.abstention_use_difficulty = abstention_use_difficulty
        self.difficulty_aware_fusion = difficulty_aware_fusion  # PHASE7
        self.calibrate_output = calibrate_output
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.calibration_min_samples = calibration_min_samples
        self.routing_mode = routing_mode
        self.n_jobs = n_jobs  # FIX 2: configurable parallelism for tree-based models/routers
        self.adaptive_n_estimators = adaptive_n_estimators  # FIX 6: opt-out flag for small-dataset n_estimators scaling
        self.bagging_mode = bagging_mode  # ACC-FIX 1
        self.feature_dropout_rate = feature_dropout_rate  # ACC-FIX 2
        self.router_holdout_fraction = router_holdout_fraction  # ACC-FIX 3
        self.combination_mode = combination_mode  # ACC-FIX 4 / ACC2-FIX 1 ("auto" value)
        self.stacking_folds = stacking_folds  # ACC-FIX 4
        self.auto_select_folds = auto_select_folds  # ACC2-FIX 1
        self.calibrate_tracks = calibrate_tracks  # ACC2-FIX 2
        self.meta_learner_cv = meta_learner_cv  # ACC2-FIX 3
        self.stacking_use_signals = stacking_use_signals  # PHASE7
        self.diversity_reweighting = diversity_reweighting  # ACC2-FIX 4
        self.router_use_oof = router_use_oof  # ACC-FIX 5
        self.small_data_auto_mode = small_data_auto_mode  # ACC-FIX 6
        self.small_data_threshold = small_data_threshold  # ACC-FIX 6
        self.router_fallback_threshold = router_fallback_threshold  # ACC-FIX 7
        self.router_fallback_blend = router_fallback_blend  # ACC-FIX 7
        self.track_early_stopping = track_early_stopping  # CB-FIX 2
        self.diversity_reweighting_folds = max(2, diversity_reweighting_folds)  # CB-FIX 3
        self.intelligent_routing = intelligent_routing
        self.loss_aware_normalization = loss_aware_normalization  # PHASE1-F1
        self.expert_feature_subsets = expert_feature_subsets      # PHASE1-F2
        self.routing_regions = routing_regions
        self.routing_bootstrap_ensemble_size = routing_bootstrap_ensemble_size
        self.routing_memory_size = routing_memory_size
        self.adaptive_top_k_max = adaptive_top_k_max
        
        # Initialize components
        self.tracks: Dict[str, Track] = {}
        self.preprocessor_ = None
        self.feature_selector_ = None
        self.fitted_ = False
        self.classes_ = None
        self.n_features_in_ = None
        self.class_weights_ = None
        self.prediction_count_ = 0
        self.router_ = None
        self.correction_track_ = None
        self.correction_gain_ = None
        self._conformal_residuals_ = None
        self._conformal_calibrated_ = False
        self._calibration_method_ = None
        self._calibration_params_ = None
        self.calibration_report_ = None
        self.expert_loss_model_ = None
        self._loss_aware_track_names_ = []
        self.difficulty_model_ = None
        self._difficulty_train_scores_ = None
        self.diversity_diagnostics_ = None
        self.kmeans_ = None
        self.signal_extractor_ = None  # SIGNAL-GUIDED ROUTING: Structural signal extraction
        self._router_uses_meta_features = False
        self._load_balance_loss_history = []
        self._dynamic_tracks_created = 0
        # PHASE4 (spec section 11/24): dynamic-spawning safe buffer, cooldown tracking, and
        # drift detection. The labeled buffer only ever gets real labels appended to it (from
        # partial_fit); it is never populated with self-generated pseudo-labels unless
        # self_training=True explicitly enables that separate, riskier path.
        self._uncertain_buffer_X_ = None
        self._uncertain_buffer_y_ = None
        self._last_spawn_step_ = -10**9
        self._call_step_ = 0
        self._drift_detector_ = PageHinkleyDetector(threshold=self.drift_threshold)
        self.drift_detected_ = False
        self._drift_event_count_ = 0
        # PHASE4 (spec item 12): the data most recently used to train the router/fusion layer,
        # retained so refresh_router()/refresh_fusion() can retrain against the CURRENT track
        # set without needing the caller to re-supply training data.
        self._last_router_X_ = None
        self._last_router_y_ = None
        # PHASE6 (spec item 27): lightweight, real (non-fabricated) counters for
        # get_diagnostics()/get_routing_report() — incremented at the actual decision points in
        # predict()/predict_proba(), not derived after the fact.
        self._abstention_count_ = 0
        self._correction_applied_count_ = 0
        self._routing_history_ = deque(maxlen=2000)  # track name selected per sample (hard-routing choice)
        self._inference_latencies_ = deque(maxlen=500)  # wall-clock seconds per predict() call
        self.meta_learner_ = None  # ACC-FIX 4: stacking meta-learner (set only when combination_mode resolves to "stacking")
        self._stacking_track_names_ = None  # ACC-FIX 4: track name order used to build the stacked meta-feature matrix
        # ACC-FIX 6: records which combination_mode was ACTUALLY used for the most recent fit()
        # (may differ from self.combination_mode when small_data_auto_mode silently switched it);
        # predict()/predict_proba() consult this instead of self.combination_mode directly so the
        # auto-selected mode is honored consistently without mutating the constructor hyperparameter.
        self._active_combination_mode_ = None
        # PHASE3-G1: risk threshold (on 1 - max fused probability) above which the correction
        # track is allowed to override, chosen by held-out simulation of the deployed rule in
        # _add_correction_track. None means "no validated operating point" -> never applies.
        self.correction_risk_threshold_ = None
        self.combination_mode_scores_ = None
        self.global_router_ = None
        self.global_router_ensemble_ = []
        self.routing_region_clusterer_ = None
        self._regional_routers_ = {}
        self._region_track_membership_ = {}
        self._region_expert_performance_ = {}
        self._router_memory_ = None
        self._routing_memory_initialized_ = False
        self._last_routing_uncertainty_ = None
        self._last_routing_entropy_ = None
        self._last_dynamic_top_k_ = None
        # PHASE0 (measurement): per-sample record of how many experts ACTUALLY contributed to
        # the fused output, measured at the fusion point rather than inferred from the top_k
        # hyperparameter (which, in the flat routing path, does not gate fusion at all).
        # Stored as the perplexity exp(H(w)) of the fusion weight vector: 1.0 under hard
        # routing, n_tracks under a perfectly uniform soft blend.
        self._effective_experts_history_ = deque(maxlen=20000)
        # PHASE0 (measurement): per-sample adaptive top-k budget as computed by
        # _compute_dynamic_top_k. Recorded separately from _effective_experts_history_ because
        # the budget and the realized fusion width are NOT the same quantity.
        self._dynamic_top_k_history_ = deque(maxlen=20000)
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_adaptive_n_estimators(self, n_samples: Optional[int]) -> int:
        """FIX 6: Scale n_estimators down for small datasets (<~2000 rows).

        Fixed-size forests/boosted ensembles waste compute on small datasets
        without meaningfully improving accuracy. Scales linearly down to a floor,
        and is a no-op (returns configured default) for larger datasets or when
        n_samples is unknown, and can be disabled via adaptive_n_estimators=False.
        """
        if not self.adaptive_n_estimators or n_samples is None:
            return self.n_estimators
        if n_samples < 2000:
            scale = max(0.3, n_samples / 2000.0)
            return max(10, int(self.n_estimators * scale))
        return self.n_estimators

    def _clone_track_model(self, estimator):
        """Clone a heterogeneous track estimator, with a CatBoost deepcopy fallback when needed."""
        try:
            return clone(estimator)
        except Exception:
            if HAS_CATBOOST and isinstance(estimator, (cb.CatBoostClassifier, cb.CatBoostRegressor)):
                # CB-FIX 1: CatBoost sometimes resists sklearn.clone in older releases, so deepcopy
                # keeps the per-track expert isolated without changing the existing rotation logic.
                return deepcopy(estimator)
            raise

    def _wrap_xgb_if_needed(self, estimator, force: bool = False):
        """PHASE4 fix: wrap an XGBoost classifier when this model's label space is not 0..k-1.

        Applied where the estimator is CONSTRUCTED rather than where it is fitted, because the
        call sites pass an estimator in and then keep their own reference to it, ignoring
        _fit_track_model's return value — so a wrapper created at fit time would be discarded.
        Uses self.classes_, which fit() establishes before any track is built.
        """
        if not (HAS_XGBOOST and isinstance(estimator, xgb.XGBClassifier)):
            return estimator
        # PHASE5: `force=True` is used for ROUTER models, whose label space is TRACK INDICES,
        # not task classes. A router's best-track labels are routinely non-contiguous — if
        # track 0 is never the best expert for any sample, the labels are [1, 2, 3] and XGBoost
        # rejects them with "Invalid classes inferred from unique values of `y`". The router
        # then fell back to a weaker model with only a WARNING, so routing quietly degraded
        # while still reporting a successful fit. self.classes_ describes the task, so it
        # cannot detect this case; wrapping unconditionally is correct and costs nothing when
        # the labels already happen to be contiguous.
        if force:
            return _LabelEncodedClassifier(estimator)
        classes = getattr(self, 'classes_', None)
        if classes is None or len(classes) == 0:
            return estimator
        classes = np.asarray(classes)
        needs_wrap = (not np.issubdtype(classes.dtype, np.integer)
                      or not np.array_equal(classes, np.arange(len(classes))))
        return _LabelEncodedClassifier(estimator) if needs_wrap else estimator

    def _fit_track_model(self, estimator, X_fit: np.ndarray, y_fit: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit a track estimator, optionally using a local early-stopping slice when supported."""
        if not self.track_early_stopping:
            fit_kwargs = {}
            if sample_weight is not None and has_fit_parameter(estimator, 'sample_weight'):
                fit_kwargs['sample_weight'] = sample_weight
            estimator.fit(X_fit, y_fit, **fit_kwargs)
            return estimator

        supported_booster = (
            (HAS_XGBOOST and isinstance(estimator, (xgb.XGBClassifier, xgb.XGBRegressor)))
            or (HAS_LIGHTGBM and isinstance(estimator, (lgb.LGBMClassifier, lgb.LGBMRegressor)))
            or (HAS_CATBOOST and isinstance(estimator, (cb.CatBoostClassifier, cb.CatBoostRegressor)))
        )
        if not supported_booster or len(X_fit) < 20:
            fit_kwargs = {}
            if sample_weight is not None and has_fit_parameter(estimator, 'sample_weight'):
                fit_kwargs['sample_weight'] = sample_weight
            estimator.fit(X_fit, y_fit, **fit_kwargs)
            return estimator

        # CB-FIX 2: carve a small local validation slice from this track's own training data so
        # boosted experts can stop early without ever touching the global router holdout split.
        val_fraction = 0.12
        split_kwargs = {'test_size': val_fraction, 'random_state': self.random_state}
        fit_sample_weight = None
        if sample_weight is not None and has_fit_parameter(estimator, 'sample_weight'):
            fit_sample_weight = sample_weight

        try:
            if self.task_type == 'classification':
                unique_classes, class_counts = np.unique(y_fit, return_counts=True)
                if len(unique_classes) > 1 and class_counts.min() >= 2:
                    split_kwargs['stratify'] = y_fit
            if sample_weight is not None:
                X_train, X_val, y_train, y_val, sw_train, _ = train_test_split(
                    X_fit, y_fit, sample_weight, **split_kwargs
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit, **split_kwargs)
                sw_train = None
        except Exception:
            fit_kwargs = {}
            if fit_sample_weight is not None:
                fit_kwargs['sample_weight'] = fit_sample_weight
            estimator.fit(X_fit, y_fit, **fit_kwargs)
            return estimator

        fit_attempts = []
        if HAS_XGBOOST and isinstance(estimator, (xgb.XGBClassifier, xgb.XGBRegressor)):
            fit_attempts = [
                {'eval_set': [(X_val, y_val)], 'early_stopping_rounds': 20, 'verbose': False},
                {'eval_set': [(X_val, y_val)], 'verbose': False},
                {},
            ]
        elif HAS_LIGHTGBM and isinstance(estimator, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            fit_attempts = []
            if hasattr(lgb, 'early_stopping'):
                fit_attempts.append({'eval_set': [(X_val, y_val)], 'callbacks': [lgb.early_stopping(20, verbose=False)]})
            fit_attempts.extend([
                {'eval_set': [(X_val, y_val)], 'early_stopping_rounds': 20},
                {'eval_set': [(X_val, y_val)]},
                {},
            ])
        elif HAS_CATBOOST and isinstance(estimator, (cb.CatBoostClassifier, cb.CatBoostRegressor)):
            fit_attempts = [
                {'eval_set': [(X_val, y_val)], 'use_best_model': True, 'early_stopping_rounds': 20, 'verbose': False},
                {'eval_set': [(X_val, y_val)], 'use_best_model': True, 'verbose': False},
                {'eval_set': [(X_val, y_val)], 'verbose': False},
                {},
            ]

        for extra_kwargs in fit_attempts:
            fit_kwargs = dict(extra_kwargs)
            if fit_sample_weight is not None:
                fit_kwargs['sample_weight'] = sw_train
            try:
                estimator.fit(X_train, y_train, **fit_kwargs)
                return estimator
            except Exception:
                continue

        fit_kwargs = {}
        if fit_sample_weight is not None:
            fit_kwargs['sample_weight'] = fit_sample_weight
        estimator.fit(X_fit, y_fit, **fit_kwargs)
        return estimator

    def _compute_diversity_probe_oof_errors(self, clf, X_track_clf: np.ndarray, y_track: np.ndarray,
                                            sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """CB-FIX 3: leak-free OOF probe errors for diversity reweighting.

        This borrows CatBoost's ordered/OOF bias-control idea in spirit: the probe only scores
        rows that were held out from its fold fit, so memorized training rows cannot make the
        diversity signal look artificially good. That is more expensive than the old single
        probe fit, but it keeps the reweighting signal honest.
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        n_local = len(X_track_clf)
        if n_local < 4:
            return np.zeros(n_local, dtype=float)

        n_splits = min(self.diversity_reweighting_folds, n_local)
        if n_splits < 2:
            return np.zeros(n_local, dtype=float)

        if self.task_type == 'classification':
            unique_classes, class_counts = np.unique(y_track, return_counts=True)
            use_stratified = len(unique_classes) > 1 and class_counts.min() >= n_splits
        else:
            use_stratified = False

        if use_stratified:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(X_track_clf, y_track)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(X_track_clf)

        local_errors = np.zeros(n_local, dtype=float)
        local_counts = np.zeros(n_local, dtype=float)

        for train_idx, val_idx in split_iter:
            probe_clf = self._clone_track_model(clf)
            X_fold_train = X_track_clf[train_idx]
            y_fold_train = y_track[train_idx]
            fold_weights = sample_weight[train_idx] if sample_weight is not None else None
            self._fit_track_model(probe_clf, X_fold_train, y_fold_train, sample_weight=fold_weights)

            X_fold_val = X_track_clf[val_idx]
            y_fold_val = y_track[val_idx]
            if self.task_type == 'classification':
                fold_preds = probe_clf.predict(X_fold_val)
                wrong = (fold_preds != y_fold_val).astype(float)
            else:
                fold_preds = probe_clf.predict(X_fold_val)
                abs_err = np.abs(fold_preds - y_fold_val)
                max_err = np.max(abs_err) + 1e-10
                wrong = abs_err / max_err

            local_errors[val_idx] += wrong
            local_counts[val_idx] += 1.0

        local_counts = np.maximum(local_counts, 1.0)
        return local_errors / local_counts

    def _initialize_router_memory(self):
        """Initialize rolling router-memory buffers used by intelligent routing.

        The memory stores recent routing confidence, uncertainty, entropy, usage, and
        region-level specialization summaries. Unlike simple counters, these deques preserve
        a short history so the router can bias toward consistently successful experts within a
        region while still adapting when the local data distribution drifts.
        """
        self._router_memory_ = {
            'global': {
                'confidence': deque(maxlen=self.routing_memory_size),
                'uncertainty': deque(maxlen=self.routing_memory_size),
                'entropy': deque(maxlen=self.routing_memory_size),
                'usage': deque(maxlen=self.routing_memory_size),
            },
            'regions': {},
            'experts': {},
        }
        self._routing_memory_initialized_ = True

    def _create_global_router(self) -> BaseEstimator:
        """Create the global router that predicts routing context / region IDs.

        This router is not asked to choose experts directly. Its job is to partition the
        input space into a small number of routing contexts so that later routing stages can
        specialize locally. That hierarchy reduces the burden on a single flat router, while
        preserving the existing router model family (XGBoost, LightGBM, CatBoost, or MLP).
        """
        return self._create_stronger_router()

    def _create_regional_router(self) -> BaseEstimator:
        """Create a regional router that predicts expert probabilities within one region.

        Regional routers are intentionally lightweight reuses of the existing router family.
        They only see samples assigned to their region, which sharpens specialization and lets
        the final routing score incorporate both local region context and historical expert
        success statistics.
        """
        return self._create_stronger_router()

    def _align_router_output(self, proba: np.ndarray, model_classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        """Align router probabilities to a target class space.

        Router models can emit only a subset of classes when trained on sparse regions.
        This helper keeps the hierarchical router architecture numerically stable by mapping
        those local columns back into the global class space with zeros for unseen classes.
        """
        target_classes = np.asarray(target_classes)
        aligned = np.zeros((proba.shape[0], len(target_classes)))
        for local_idx, cls in enumerate(model_classes):
            matches = np.where(target_classes == cls)[0]
            if len(matches) > 0:
                aligned[:, matches[0]] = proba[:, local_idx]
        return aligned

    def _compute_uncertainty(self, mean_proba: np.ndarray, ensemble_probas: Optional[np.ndarray] = None,
                             signal_disagreement: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate router uncertainty from entropy and router disagreement.

        The method returns both normalized entropy and a bounded uncertainty score. When a
        bootstrap router ensemble is available, the ensemble variance acts as an epistemic
        uncertainty proxy; otherwise the entropy term and signal disagreement provide a cheap
        fallback. This keeps the implementation compatible with the current production router
        while still exposing per-sample uncertainty to the new routing score.
        """
        clipped = np.clip(mean_proba, 1e-10, 1.0)
        clipped = clipped / clipped.sum(axis=1, keepdims=True)
        entropy = -np.sum(clipped * np.log(clipped + 1e-10), axis=1)
        entropy = entropy / np.log(clipped.shape[1] + 1e-10)
        entropy = np.clip(entropy, 0.0, 1.0)

        variance_term = np.zeros(len(mean_proba))
        if ensemble_probas is not None and len(ensemble_probas) > 1:
            variance_term = np.var(ensemble_probas, axis=0).mean(axis=1)
            variance_term = variance_term / (variance_term.max() + 1e-10)

        disagreement_term = np.zeros(len(mean_proba))
        if signal_disagreement is not None:
            disagreement_term = np.asarray(signal_disagreement, dtype=float)
            disagreement_term = disagreement_term / (disagreement_term.max() + 1e-10)

        uncertainty = 0.5 * entropy + 0.3 * variance_term + 0.2 * disagreement_term
        uncertainty = np.clip(uncertainty, 0.0, 1.0)
        return entropy, uncertainty

    def _compute_dynamic_top_k(self, confidence: np.ndarray, uncertainty: np.ndarray,
                               signal_disagreement: np.ndarray) -> np.ndarray:
        """Compute a per-sample adaptive top-k budget for expert selection.

        Easy samples stay near top-1, while hard or highly uncertain samples expand toward the
        configured ceiling. This preserves low latency on the bulk of routine inputs while
        allowing the router to widen its candidate set when uncertainty or expert disagreement
        indicates that the sample is likely to sit near a decision boundary.
        """
        if not getattr(self, 'intelligent_routing', False):
            return np.full(len(confidence), min(self.top_k, self.n_tracks), dtype=int)

        confidence = np.clip(confidence, 0.0, 1.0)
        uncertainty = np.clip(uncertainty, 0.0, 1.0)
        signal_disagreement = np.clip(signal_disagreement, 0.0, 1.0)

        risk = 0.35 * (1.0 - confidence) + 0.35 * uncertainty + 0.30 * signal_disagreement
        max_k = min(self.n_tracks, self.adaptive_top_k_max)
        dynamic_k = np.ones(len(confidence), dtype=int)
        dynamic_k[risk >= 0.20] = min(2, max_k)
        dynamic_k[risk >= 0.40] = min(4, max_k)
        dynamic_k[risk >= 0.65] = max_k
        return np.clip(dynamic_k, 1, max_k)

    def _update_region_statistics(self, region_ids: np.ndarray, chosen_track_indices: np.ndarray,
                                  routing_confidences: np.ndarray, uncertainty: np.ndarray,
                                  top_k: np.ndarray, y_true: Optional[np.ndarray] = None,
                                  track_output_cache: Optional[dict] = None):
        """Update rolling region and expert statistics after a labeled routing pass.

        When labels are available we record actual expert success rates in the current region.
        That history is later folded into the routing score so the model can prefer experts that
        have repeatedly performed well in the same region instead of re-discovering that fact on
        every prediction batch.
        """
        if not self._routing_memory_initialized_:
            self._initialize_router_memory()

        for region_id in np.unique(region_ids):
            mask = region_ids == region_id
            region_key = int(region_id)
            if region_key not in self._router_memory_['regions']:
                self._router_memory_['regions'][region_key] = {
                    'confidence': deque(maxlen=self.routing_memory_size),
                    'uncertainty': deque(maxlen=self.routing_memory_size),
                    'entropy': deque(maxlen=self.routing_memory_size),
                    'accuracy': deque(maxlen=self.routing_memory_size),
                    'usage': deque(maxlen=self.routing_memory_size),
                }
            region_memory = self._router_memory_['regions'][region_key]
            region_memory['confidence'].append(float(np.mean(routing_confidences[mask])))
            region_memory['uncertainty'].append(float(np.mean(uncertainty[mask])))
            region_memory['usage'].append(int(mask.sum()))
            if y_true is not None:
                region_memory['accuracy'].append(float(np.mean(chosen_track_indices[mask] >= 0)))

        self._router_memory_['global']['confidence'].append(float(np.mean(routing_confidences)))
        self._router_memory_['global']['uncertainty'].append(float(np.mean(uncertainty)))
        self._router_memory_['global']['usage'].append(int(len(region_ids)))

        if y_true is None or track_output_cache is None:
            return

        track_names = [t for t in self.tracks if t != 'correction_track']
        for region_id, router_spec in self._regional_routers_.items():
            member_track_indices = router_spec.get('track_indices', [])
            if len(member_track_indices) == 0:
                continue
            region_mask = region_ids == region_id
            if not np.any(region_mask):
                continue
            for track_idx in member_track_indices:
                track_name = track_names[track_idx] if track_idx < len(track_names) else None
                if track_name is None or track_name not in track_output_cache:
                    continue
                if track_name not in self._router_memory_['experts']:
                    self._router_memory_['experts'][track_name] = {
                        'performance': deque(maxlen=self.routing_memory_size),
                        'confidence': deque(maxlen=self.routing_memory_size),
                        'uncertainty': deque(maxlen=self.routing_memory_size),
                        'usage': deque(maxlen=self.routing_memory_size),
                    }
                expert_memory = self._router_memory_['experts'][track_name]
                expert_memory['confidence'].append(float(np.mean(routing_confidences[region_mask])))
                expert_memory['uncertainty'].append(float(np.mean(uncertainty[region_mask])))
                expert_memory['usage'].append(int(region_mask.sum()))

    def _update_router_memory(self, region_ids: np.ndarray, chosen_track_indices: np.ndarray,
                              routing_confidences: np.ndarray, uncertainty: np.ndarray,
                              top_k: np.ndarray):
        """Update rolling memory during inference with no labels available.

        This captures the recent routing context, the confidence distribution, and the dynamic
        top-k budget so later predictions can bias toward stable regions and avoid repeatedly
        over-routing uncertain samples through the same brittle path.
        """
        if not self._routing_memory_initialized_:
            self._initialize_router_memory()

        for region_id in np.unique(region_ids):
            mask = region_ids == region_id
            region_key = int(region_id)
            if region_key not in self._router_memory_['regions']:
                self._router_memory_['regions'][region_key] = {
                    'confidence': deque(maxlen=self.routing_memory_size),
                    'uncertainty': deque(maxlen=self.routing_memory_size),
                    'entropy': deque(maxlen=self.routing_memory_size),
                    'accuracy': deque(maxlen=self.routing_memory_size),
                    'usage': deque(maxlen=self.routing_memory_size),
                }
            region_memory = self._router_memory_['regions'][region_key]
            region_memory['confidence'].append(float(np.mean(routing_confidences[mask])))
            region_memory['uncertainty'].append(float(np.mean(uncertainty[mask])))
            region_memory['usage'].append(int(mask.sum()))

        self._router_memory_['global']['confidence'].append(float(np.mean(routing_confidences)))
        self._router_memory_['global']['uncertainty'].append(float(np.mean(uncertainty)))
        self._router_memory_['global']['usage'].append(int(len(region_ids)))

    def _route_batch(self, X_selected: np.ndarray, task_type: Optional[str] = None,
                     track_output_cache: Optional[dict] = None) -> Dict[str, Any]:
        """Route a batch through the hierarchical router and return routing artifacts.

        The batch router is shared by predict() and predict_proba() so the global router,
        regional routers, uncertainty estimates, and dynamic top-k decisions are computed once
        and reused across downstream prediction steps. This is the main latency safeguard for
        the hierarchical routing path.
        """
        if task_type is None:
            task_type = self.task_type
        if track_output_cache is None:
            track_output_cache = {}

        track_names = [t for t in self.router_track_names_ if t in self.tracks and t != 'correction_track']
        n_samples = len(X_selected)
        n_tracks = len(track_names)

        if n_tracks == 0:
            return {
                'region_ids': np.zeros(n_samples, dtype=int),
                'track_weights': np.zeros((n_samples, 0)),
                'routing_confidences': np.ones(n_samples),
                'uncertainty': np.zeros(n_samples),
                'entropy': np.zeros(n_samples),
                'dynamic_top_k': np.ones(n_samples, dtype=int),
                'track_output_cache': track_output_cache,
            }

        # Compute per-track outputs once and reuse them for signals, scoring, and final prediction.
        track_series = []
        track_confidences = []
        for track_name in track_names:
            track = self.tracks[track_name]
            if track_name in track_output_cache:
                track_output = track_output_cache[track_name]
            else:
                if task_type == 'classification':
                    track_output = track.predict_proba(X_selected)
                else:
                    track_output = track.predict(X_selected).astype(float)
                track_output_cache[track_name] = track_output
            if task_type == 'classification':
                track_pred = np.argmax(track_output, axis=1)
                track_conf = np.max(track_output, axis=1)
            else:
                track_pred = np.asarray(track_output, dtype=float)
                track_conf = np.ones(n_samples)
            track_series.append(track_pred)
            track_confidences.append(track_conf)

        if task_type == 'classification':
            track_matrix = np.column_stack(track_series)
            signal_disagreement = np.std(track_matrix, axis=1)
            signal_disagreement = signal_disagreement / (np.max(signal_disagreement) + 1e-10)
        else:
            track_matrix = np.column_stack(track_series)
            signal_disagreement = np.std(track_matrix, axis=1)
            signal_disagreement = signal_disagreement / (np.max(signal_disagreement) + 1e-10)

        X_for_router = X_selected
        if getattr(self, '_router_uses_meta_features', False):
            X_for_router = self._build_router_meta_features(X_selected, track_output_cache=track_output_cache)

        global_prob_members = []
        global_router_members = self.global_router_ensemble_ if getattr(self, 'intelligent_routing', False) and self.global_router_ensemble_ else [self.router_]
        target_region_classes = np.arange(getattr(self, '_n_routing_regions_', getattr(self, 'routing_regions', 1)))
        for router_model in global_router_members:
            if router_model is None:
                continue
            member_proba = router_model.predict_proba(X_for_router)
            member_classes = getattr(router_model, 'classes_', np.arange(member_proba.shape[1]))
            aligned = self._align_router_output(member_proba, member_classes, target_region_classes)
            global_prob_members.append(aligned)

        if not global_prob_members:
            raise ValueError("No global router available for intelligent routing")

        global_probs_mean = np.mean(global_prob_members, axis=0)
        global_probs_mean = global_probs_mean / (global_probs_mean.sum(axis=1, keepdims=True) + 1e-10)
        global_probs_stack = np.stack(global_prob_members, axis=0)
        global_entropy, global_uncertainty = self._compute_uncertainty(
            global_probs_mean, ensemble_probas=global_probs_stack, signal_disagreement=signal_disagreement
        )

        region_ids = np.argmax(global_probs_mean, axis=1)
        expert_scores = np.zeros((n_samples, n_tracks))

        for region_id in np.unique(region_ids):
            sample_mask = region_ids == region_id
            if not np.any(sample_mask):
                continue
            region_spec = getattr(self, '_regional_routers_', {}).get(int(region_id))
            if region_spec is None:
                continue

            region_router = region_spec['router']
            region_track_indices = np.asarray(region_spec['track_indices'], dtype=int)
            if len(region_track_indices) == 0:
                continue
            region_proba = region_router.predict_proba(X_for_router[sample_mask])
            if region_proba.ndim == 1:
                region_proba = region_proba.reshape(-1, 1)
            region_proba_full = np.zeros((region_proba.shape[0], n_tracks))
            # IMPROVEMENT: Map sparse regional labels before expanding to the global track pool.
            region_classes = getattr(region_router, 'classes_', np.arange(region_proba.shape[1]))
            for column, local_idx in enumerate(region_classes):
                if 0 <= int(local_idx) < len(region_track_indices):
                    region_proba_full[:, region_track_indices[int(local_idx)]] = region_proba[:, column]
            region_proba = region_proba_full[:, region_track_indices]

            region_history = self._region_expert_performance_.get(int(region_id), {})
            region_history_scores = np.ones(len(region_track_indices))
            for idx, track_idx in enumerate(region_track_indices):
                track_name = track_names[track_idx] if track_idx < len(track_names) else None
                region_history_scores[idx] = float(region_history.get(track_name, 1.0)) if track_name is not None else 1.0

            sample_conf = np.max(region_proba, axis=1)
            sample_uncertainty = global_uncertainty[sample_mask]
            sample_entropy = global_entropy[sample_mask]
            sample_top_k = self._compute_dynamic_top_k(sample_conf, sample_uncertainty, signal_disagreement[sample_mask])

            for row_pos, sample_idx in enumerate(np.where(sample_mask)[0]):
                local_scores = region_proba[row_pos] * region_history_scores
                local_scores = local_scores * (1.0 - sample_uncertainty[row_pos])
                local_scores = local_scores * (1.0 - sample_entropy[row_pos])
                local_scores = local_scores * (1.0 - signal_disagreement[sample_idx])
                if np.all(local_scores <= 0):
                    local_scores = np.ones_like(local_scores)
                k = int(sample_top_k[row_pos])
                top_local = np.argsort(local_scores)[::-1][:k]
                chosen_local_scores = local_scores[top_local]
                if chosen_local_scores.sum() <= 0:
                    chosen_local_scores = np.ones_like(chosen_local_scores)
                chosen_local_scores = chosen_local_scores / chosen_local_scores.sum()
                for local_rank, local_idx in enumerate(top_local):
                    global_track_idx = region_track_indices[local_idx]
                    expert_scores[sample_idx, global_track_idx] = chosen_local_scores[local_rank] * global_probs_mean[sample_idx, int(region_id)]

        row_sums = expert_scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        track_weights = expert_scores / row_sums

        routing_confidences = np.max(track_weights, axis=1)
        if np.all(routing_confidences == 0):
            routing_confidences = 1.0 - global_uncertainty

        chosen_track_indices = np.argmax(track_weights, axis=1)
        dynamic_top_k = self._compute_dynamic_top_k(routing_confidences, global_uncertainty, signal_disagreement)

        self._last_routing_uncertainty_ = global_uncertainty
        self._last_routing_entropy_ = global_entropy
        self._last_dynamic_top_k_ = dynamic_top_k
        self._dynamic_top_k_history_.extend(np.asarray(dynamic_top_k).astype(int).tolist())  # PHASE0

        self._update_router_memory(region_ids, chosen_track_indices, routing_confidences, global_uncertainty, dynamic_top_k)

        return {
            'region_ids': region_ids,
            'track_weights': track_weights,
            'routing_confidences': routing_confidences,
            'uncertainty': global_uncertainty,
            'entropy': global_entropy,
            'dynamic_top_k': dynamic_top_k,
            'track_output_cache': track_output_cache,
            'chosen_track_indices': chosen_track_indices,
            'global_region_probabilities': global_probs_mean,
        }

    def _train_hierarchical_router(self, X_holdout: np.ndarray, y_holdout: np.ndarray,
                                   best_track_indices_override: Optional[np.ndarray] = None,
                                   oof_out: Optional[np.ndarray] = None):
        """Train the hierarchical routing stack used by TRA 2.0.

        The stack first learns routing regions, then learns a regional router for each region,
        and finally uses region-specific expert-history priors to rank the experts inside each
        region. That keeps the routing problem local and reduces the burden on a single flat
        router without touching expert learning, stacking, or correction logic.

        PHASE3-FIX (spec item I): when `oof_out` is provided (the router_use_oof path, where
        X_holdout is NOT a genuine holdout — it's X_train_tracks), every per-track evaluation
        in this method — the global meta-features, the per-region "which track is best here"
        scores, and the regional router's own training labels — is now sourced from these
        leak-free OOF outputs instead of calling track.predict_proba(X_holdout)/
        track.predict(X_holdout) directly. The ORIGINAL implementation only used the OOF
        override for the (barely-consumed) `best_track_indices` bookkeeping variable while the
        actual region-membership scores and regional-router training labels — the two things
        that most directly shape hierarchical routing behavior — were still computed in-sample
        on tracks' own training rows, silently defeating the point of router_use_oof for
        intelligent_routing specifically. When oof_out is None (genuine holdout), behavior is
        byte-identical to before this fix.
        """
        logger.info(f"Training hierarchical {self.router_type} router stack...")

        track_names = [t for t in self.tracks.keys() if t != 'correction_track']
        n_tracks = len(track_names)
        n_holdout = len(X_holdout)
        if n_tracks == 0:
            self.router_ = None
            self.global_router_ = None
            self._regional_routers_ = {}
            return

        track_output_cache = {}
        if oof_out is not None:
            # PHASE3-FIX: seed from leak-free OOF outputs (see docstring above).
            for i, tname in enumerate(track_names[:oof_out.shape[1]]):
                track_output_cache[tname] = oof_out[:, i, :] if self.task_type == "classification" else oof_out[:, i]

        if best_track_indices_override is not None:
            best_track_indices = best_track_indices_override
        elif self.task_type == "classification":
            track_probas = np.zeros((n_holdout, n_tracks))
            true_class_indices = np.array([
                np.where(self.classes_ == y_holdout[j])[0][0]
                if y_holdout[j] in self.classes_ else 0
                for j in range(n_holdout)
            ])
            for i, track_name in enumerate(track_names):
                track = self.tracks[track_name]
                try:
                    if track_name in track_output_cache:
                        proba = track_output_cache[track_name]
                    else:
                        proba = track.predict_proba(X_holdout)
                        track_output_cache[track_name] = proba
                    # PHASE4-FIX: same fix as _train_router — align before indexing so a track
                    # that never saw every class during its own fit (small buffer subsets from
                    # dynamic spawning are especially prone to this) can't raise an
                    # out-of-bounds error here.
                    track_classes = getattr(track.classifier, 'classes_', self.classes_)
                    proba_aligned = self._align_proba(proba, track_classes)
                    track_probas[:, i] = proba_aligned[np.arange(n_holdout), true_class_indices]
                except Exception as e:
                    logger.warning(f"Failed to evaluate track {track_name}: {e}")
                    track_probas[:, i] = -1.0
            best_track_indices = np.argmax(track_probas, axis=1)
        else:
            track_errors = np.zeros((n_holdout, n_tracks))
            for i, track_name in enumerate(track_names):
                track = self.tracks[track_name]
                try:
                    if track_name in track_output_cache:
                        preds = track_output_cache[track_name]
                    else:
                        preds = track.predict(X_holdout).astype(float)
                        track_output_cache[track_name] = preds
                    track_errors[:, i] = np.abs(preds - y_holdout)
                except Exception as e:
                    logger.warning(f"Failed to evaluate track {track_name}: {e}")
                    track_errors[:, i] = np.inf
            best_track_indices = np.argmin(track_errors, axis=1)

        X_router_train = self._build_router_meta_features(X_holdout, track_output_cache=track_output_cache)
        self._router_uses_meta_features = (X_router_train.shape[1] > X_holdout.shape[1])
        self._initialize_router_memory()

        n_regions = min(self.routing_regions, max(2, n_holdout // 20), max(2, n_tracks))
        self._n_routing_regions_ = n_regions
        self.routing_region_clusterer_ = KMeans(n_clusters=n_regions, random_state=self.random_state)
        region_ids = self.routing_region_clusterer_.fit_predict(X_router_train)

        self.global_router_ensemble_ = []
        for _ in range(self.routing_bootstrap_ensemble_size):
            router = self._create_global_router()
            boot_idx = np.random.choice(n_holdout, size=n_holdout, replace=True)
            try:
                router.fit(X_router_train[boot_idx], region_ids[boot_idx])
            except Exception:
                router.fit(X_router_train, region_ids)
            self.global_router_ensemble_.append(router)
        self.global_router_ = self.global_router_ensemble_[0] if self.global_router_ensemble_ else self._create_global_router()
        self.router_ = self.global_router_

        self._regional_routers_ = {}
        self._region_track_membership_ = {}
        self._region_expert_performance_ = {}
        for region_id in range(n_regions):
            region_mask = region_ids == region_id
            if not np.any(region_mask):
                continue

            region_sample_idx = np.where(region_mask)[0]
            region_track_scores = np.zeros(n_tracks)
            for track_idx, track_name in enumerate(track_names):
                try:
                    if self.task_type == "classification":
                        proba = track_output_cache.get(track_name)
                        if proba is None:
                            proba = self.tracks[track_name].predict_proba(X_holdout)
                        # PHASE4-FIX: align before indexing (see the two fixes above in this
                        # same method) — a track that never saw every class during its own fit
                        # would otherwise raise an out-of-bounds error here too.
                        track_classes = getattr(self.tracks[track_name].classifier, 'classes_', self.classes_)
                        proba_aligned = self._align_proba(proba, track_classes)
                        true_class_indices = np.array([
                            np.where(self.classes_ == y_holdout[j])[0][0]
                            if y_holdout[j] in self.classes_ else 0
                            for j in region_sample_idx
                        ])
                        region_track_scores[track_idx] = float(np.mean(proba_aligned[region_sample_idx, true_class_indices]))
                    else:
                        preds = track_output_cache.get(track_name)
                        if preds is None:
                            preds = self.tracks[track_name].predict(X_holdout).astype(float)
                        region_track_scores[track_idx] = float(1.0 / (1.0 + np.mean(np.abs(preds[region_sample_idx] - y_holdout[region_mask]))))
                except Exception:
                    region_track_scores[track_idx] = 0.0

            region_track_indices = np.argsort(region_track_scores)[::-1][:min(max(2, self.top_k), n_tracks)]
            self._region_track_membership_[region_id] = region_track_indices.tolist()

            region_router_labels = []
            for sample_idx in region_sample_idx:
                best_local_idx = 0
                best_score = -np.inf if self.task_type == "classification" else np.inf
                for local_idx, track_idx in enumerate(region_track_indices):
                    track_name = track_names[track_idx]
                    if self.task_type == "classification":
                        proba = track_output_cache.get(track_name)
                        if proba is None:
                            proba = self.tracks[track_name].predict_proba(X_holdout[[sample_idx]])
                        true_class_idx = np.where(self.classes_ == y_holdout[sample_idx])[0][0] if y_holdout[sample_idx] in self.classes_ else 0
                        score = float(proba[sample_idx, true_class_idx]) if proba.ndim == 2 and proba.shape[0] > sample_idx else float(proba[0, true_class_idx])
                        if score > best_score:
                            best_score = score
                            best_local_idx = int(local_idx)
                    else:
                        preds = track_output_cache.get(track_name)
                        if preds is None:
                            preds = self.tracks[track_name].predict(X_holdout[[sample_idx]]).astype(float)
                        err = float(np.abs(preds[sample_idx] - y_holdout[sample_idx])) if preds.ndim == 1 and len(preds) > sample_idx else float(np.abs(preds[0] - y_holdout[sample_idx]))
                        if err < best_score:
                            best_score = err
                            best_local_idx = int(local_idx)
                region_router_labels.append(best_local_idx)

            region_router = self._create_regional_router()
            try:
                region_router.fit(X_router_train[region_mask], np.asarray(region_router_labels, dtype=int))
            except Exception as e:
                logger.warning(f"Regional router fit failed for region {region_id}: {e}. Using constant fallback.")
                from sklearn.dummy import DummyClassifier
                region_router = DummyClassifier(strategy='constant', constant=0)
                region_router.fit(X_router_train[region_mask], np.asarray(region_router_labels, dtype=int))

            self._regional_routers_[region_id] = {
                'router': region_router,
                'track_indices': region_track_indices.tolist(),
                'track_names': [track_names[idx] for idx in region_track_indices],
                'n_samples': int(region_mask.sum()),
            }
            self._region_expert_performance_[region_id] = {
                track_names[idx]: float(region_track_scores[idx]) for idx in region_track_indices
            }

        self.router_track_names_ = track_names
        self._update_region_statistics(region_ids, best_track_indices, np.ones(n_holdout), np.zeros(n_holdout),
                                       np.ones(n_holdout, dtype=int), y_true=y_holdout,
                                       track_output_cache=track_output_cache)

    def _create_heterogeneous_models(self, n_samples: Optional[int] = None) -> List:
        """IMPROVEMENT 2: Create heterogeneous expert models."""
        if self.track_models is not None:
            return self.track_models
        
        models = []
        n_est = self._get_adaptive_n_estimators(n_samples)  # FIX 6: adapt n_estimators to dataset size
        
        # Model 1: Random Forest
        if self.task_type == "classification":
            models.append(RandomForestClassifier(
                n_estimators=n_est, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state, n_jobs=self.n_jobs  # FIX 2: use configurable n_jobs instead of hardcoded 1
            ))
        else:
            models.append(RandomForestRegressor(
                n_estimators=n_est, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state, n_jobs=self.n_jobs  # FIX 2: use configurable n_jobs instead of hardcoded 1
            ))
        
        # Model 2: LightGBM (if available)
        if HAS_LIGHTGBM:
            if self.task_type == "classification":
                models.append(lgb.LGBMClassifier(
                    n_estimators=n_est, max_depth=self.max_depth,
                    random_state=self.random_state, verbose=-1, n_jobs=self.n_jobs  # FIX 2
                ))
            else:
                models.append(lgb.LGBMRegressor(
                    n_estimators=n_est, max_depth=self.max_depth,
                    random_state=self.random_state, verbose=-1, n_jobs=self.n_jobs  # FIX 2
                ))
        
        # Model 3: XGBoost (if available)
        if HAS_XGBOOST:
            if self.task_type == "classification":
                models.append(self._wrap_xgb_if_needed(xgb.XGBClassifier(
                    n_estimators=n_est, max_depth=self.max_depth,
                    random_state=self.random_state, verbosity=0, n_jobs=self.n_jobs  # FIX 2
                )))
            else:
                models.append(xgb.XGBRegressor(
                    n_estimators=n_est, max_depth=self.max_depth,
                    random_state=self.random_state, verbosity=0, n_jobs=self.n_jobs  # FIX 2
                ))
        
        # Model 4: SVM (Support Vector Machine) - For heterogeneous expertise
        from sklearn.svm import SVC, SVR
        from sklearn.pipeline import Pipeline
        
        if self.task_type == "classification":
            # FIX 1: probability=True previously forced SVC's own internal 5-fold Platt-scaling CV
            # (expensive). predict_proba is used pervasively downstream (meta-features, routing,
            # aggregation) so we can't just drop probabilities; instead we calibrate explicitly with
            # CalibratedClassifierCV using fewer folds (cv=3) for a cheaper but still-valid predict_proba.
            base_svm = SVC(kernel='rbf', C=1.0, random_state=self.random_state, probability=False)
            models.append(Pipeline([
                ('scaler', StandardScaler()),
                ('svm', CalibratedClassifierCV(base_svm, cv=3))
            ]))
        else:
            models.append(Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVR(kernel='rbf', C=1.0))
            ]))
        
        # Model 5: Neural Network (if needed)
        if len(models) < self.n_tracks:
            if self.task_type == "classification":
                models.append(MLPClassifier(
                    hidden_layer_sizes=(100, 50), max_iter=200,
                    random_state=self.random_state
                ))
            else:
                models.append(MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=200,
                    random_state=self.random_state
                ))

        # CB-FIX 1: add CatBoost as the 6th heterogeneous track family, inspired by CatBoost's
        # boosted-tree expert design; we keep the existing five-track ordering intact so current
        # defaults behave the same and CatBoost only enters the rotation when extra tracks are needed.
        if HAS_CATBOOST and len(models) < self.n_tracks:
            if self.task_type == "classification":
                models.append(cb.CatBoostClassifier(
                    iterations=n_est, depth=self.max_depth, random_state=self.random_state, verbose=False
                ))
            else:
                models.append(cb.CatBoostRegressor(
                    iterations=n_est, depth=self.max_depth, random_state=self.random_state, verbose=False
                ))
        
        # PHASE1-F3: cycle the ORIGINAL pool, with a hyperparameter variation per duplicate.
        #
        # The previous line was `models.append(models[len(models) % len(models)])`, where the
        # modulus is taken against the list's CURRENT length and therefore always evaluates to
        # models[0]. Every track beyond the number of available model families became another
        # copy of the first estimator instead of cycling through the pool. Latent at n_tracks=4
        # with the optional boosting libraries installed; active whenever n_tracks exceeds the
        # number of families actually available (the spec's 5-8 track configurations, or any
        # environment missing XGBoost/LightGBM/CatBoost).
        #
        # Duplicates additionally receive a deterministic depth/capacity perturbation so that a
        # repeated FAMILY is at least not a repeated MODEL — a duplicate with identical
        # hyperparameters and identical training data contributes nothing but inference cost.
        base_pool = list(models)
        n_base = len(base_pool)
        while len(models) < self.n_tracks and n_base > 0:
            duplicate_round = len(models) // n_base
            source = base_pool[len(models) % n_base]
            models.append(self._vary_model_hyperparameters(source, duplicate_round))

        return models[:self.n_tracks]

    def _vary_model_hyperparameters(self, model, variation: int):
        """PHASE1-F3: return a clone of `model` with a deterministic hyperparameter shift.

        Used only when the pool must be padded because n_tracks exceeds the number of available
        model families. The shift is applied through get_params/set_params, so it works for any
        sklearn-compatible estimator and silently skips parameters an estimator does not expose
        (an SVM pipeline has no max_depth; a RandomForest has no C). `variation` is the
        duplication round, so the second copy of a family differs from the third.

        This is diversity by hyperparameter perturbation, which is weaker than diversity by
        model family — it is a fallback for an over-provisioned n_tracks, not a substitute for
        installing more expert families.
        """
        try:
            clone_model = self._clone_track_model(model)
        except Exception:
            return model
        try:
            params = clone_model.get_params()
        except Exception:
            return clone_model

        updates = {}
        step = int(variation)
        for key, value in params.items():
            leaf = key.rsplit('__', 1)[-1]
            if leaf == 'max_depth' and isinstance(value, (int, np.integer)) and value > 2:
                updates[key] = int(max(2, value - step))
            elif leaf == 'depth' and isinstance(value, (int, np.integer)) and value > 2:
                updates[key] = int(max(2, value - step))
            elif leaf in ('n_estimators', 'iterations') and isinstance(value, (int, np.integer)):
                updates[key] = int(max(10, int(value * (0.75 ** step))))
            elif leaf == 'C' and isinstance(value, (float, int)) and not isinstance(value, bool):
                updates[key] = float(value) * (2.0 ** step)
            elif leaf == 'random_state' and isinstance(value, (int, np.integer)):
                updates[key] = int(value) + 977 * step
        if updates:
            try:
                clone_model.set_params(**updates)
            except Exception:
                pass
        return clone_model
    
    def _get_adaptive_router_depth(self, n_samples: Optional[int], default_depth: int = 5) -> int:
        """PHASE7: shrink the router's max_depth when its training set is small.

        The router solves a proxy classification problem ("which track wins here") on the same
        row count as everything else, but that proxy target has much less real structure to
        learn than the underlying task -- a fixed 100-tree, depth-5 booster choosing between
        (typically) 4 tracks, trained on ~150-200 rows (e.g. Sonar n=208, Glass n=214), is a
        real overfitting risk with no existing safeguard: the expert tracks already have
        _get_adaptive_n_estimators, the router never had an equivalent.

        Deliberately NOT gated behind adaptive_n_estimators and deliberately reduces DEPTH
        rather than tree count: PHASE5 already established that shrinking EXPERT capacity on
        small data hurts more than it helps (the experts model the real task and need what
        capacity they can get) -- that finding does not transfer to the router's much simpler,
        much more overfitting-prone proxy target, so this is intentionally a separate mechanism
        rather than reusing that flag's (opposite) conclusion. Tree count is left alone because
        more shallow trees still costs little and preserves the ensemble-averaging benefit;
        depth is the dimension most directly responsible for a booster memorizing a small proxy
        target.
        """
        if n_samples is None:
            return default_depth
        if n_samples < 150:
            return max(2, default_depth - 2)
        if n_samples < 400:
            return max(3, default_depth - 1)
        return default_depth

    def _create_stronger_router(self, n_samples: Optional[int] = None) -> BaseEstimator:
        """IMPROVEMENT 1: Create a stronger router model.

        PHASE7: accepts an optional `n_samples` (the router's actual training-set size) used
        only to adapt max_depth downward on small datasets -- see _get_adaptive_router_depth.
        Omitting it (the hierarchical-routing call sites still do) reproduces the exact
        pre-PHASE7 fixed depth=5.
        """
        depth = self._get_adaptive_router_depth(n_samples)
        if self.router_type == "xgboost" and HAS_XGBOOST:
            return self._wrap_xgb_if_needed(xgb.XGBClassifier(
                n_estimators=100, max_depth=depth, learning_rate=0.1,
                random_state=self.random_state, verbosity=0, n_jobs=self.n_jobs  # FIX 2: use configurable n_jobs instead of hardcoded 1
            ), force=True)
        elif self.router_type == "catboost" and HAS_CATBOOST:
            return cb.CatBoostClassifier(
                iterations=100, depth=depth, learning_rate=0.1,
                random_state=self.random_state, verbose=False
            )
        elif self.router_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=200,
                random_state=self.random_state
            )
        elif self.router_type == "lightgbm" and HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                n_estimators=100, max_depth=depth, learning_rate=0.1,
                random_state=self.random_state, verbose=-1, n_jobs=self.n_jobs  # FIX 2: use configurable n_jobs instead of hardcoded 1
            )
        else:
            # PHASE1-FIX: graceful degradation (spec item 10) — the router_type requested
            # (or the "lightgbm" default) isn't available as an installed optional dependency.
            # Previously this branch unconditionally called lgb.LGBMClassifier(...) even when
            # HAS_LIGHTGBM was False, crashing with a NameError instead of degrading. Fall back
            # through the strongest available option, and finally to a RandomForest router,
            # which has no optional-dependency requirement at all.
            if HAS_LIGHTGBM:
                return lgb.LGBMClassifier(
                    n_estimators=100, max_depth=depth, learning_rate=0.1,
                    random_state=self.random_state, verbose=-1, n_jobs=self.n_jobs
                )
            elif HAS_XGBOOST:
                logger.warning(f"router_type='{self.router_type}' unavailable (missing optional "
                                f"dependency); falling back to XGBoost router.")
                return self._wrap_xgb_if_needed(xgb.XGBClassifier(
                    n_estimators=100, max_depth=depth, learning_rate=0.1,
                    random_state=self.random_state, verbosity=0, n_jobs=self.n_jobs
                ), force=True)
            elif HAS_CATBOOST:
                logger.warning(f"router_type='{self.router_type}' unavailable (missing optional "
                                f"dependency); falling back to CatBoost router.")
                return cb.CatBoostClassifier(
                    iterations=100, depth=depth, learning_rate=0.1,
                    random_state=self.random_state, verbose=False
                )
            else:
                logger.warning(f"router_type='{self.router_type}' unavailable and no optional "
                                f"gradient-boosting router dependency installed; falling back to "
                                f"a RandomForest router (no extra dependencies required).")
                return RandomForestClassifier(
                    n_estimators=100, max_depth=depth, random_state=self.random_state, n_jobs=self.n_jobs
                )
    
    def _get_base_estimator(self):
        """DEPRECATED: Use _create_heterogeneous_models instead.

        NOTE (FIX 5): still intentionally returns a single fixed model type — it's
        only used for the single residual correction track (not for the multiple
        dynamically-spawned/streamed tracks, which now cycle through
        _create_heterogeneous_models() directly at their call sites instead of
        going through this method; see partial_fit and _check_spawn_new_track).
        """
        return self._create_heterogeneous_models()[0]
            
    def _setup_preprocessing(self, X):
        """Set up automated missing value and categorical feature handling."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OrdinalEncoder
        
        if isinstance(X, pd.DataFrame):
            numeric_cols = list(X.select_dtypes(include=['number']).columns)
            cat_cols = list(X.select_dtypes(exclude=['number']).columns)
            
            transformers = []
            if numeric_cols:
                num_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', num_transformer, numeric_cols))
                
            if cat_cols:
                cat_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ])
                transformers.append(('cat', cat_transformer, cat_cols))
                
            self.preprocessor_ = ColumnTransformer(transformers=transformers)
        else:
            self.preprocessor_ = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
    
    def _setup_feature_selection(self, X: np.ndarray, y: np.ndarray):
        """Setup feature selection with adaptive parameters.
        
        IMPROVEMENT 1: Adaptive Feature Selection.
        Previously used a fixed 1/3 rule which was too aggressive (e.g., on a 20-feature dataset,
        only 6 features were kept, discarding critical signal). Now uses a 60% adaptive formula,
        and disables per-track random feature dropout when cluster_experts=True to prevent
        double-diversity mechanisms from starving the estimators.
        """
        if not self.feature_selection:
            return
        
        # Adaptive - keep 60% of features, minimum 5
        n_features = min(X.shape[1], max(5, int(X.shape[1] * 0.6)))
        method = getattr(self, 'feature_selection_method', 'model_based')
        is_clf = self.task_type == "classification"

        if method == "univariate":
            # Original behaviour. Ranks by GLOBAL univariate association, which systematically
            # discards regime-conditional features (see the feature_selection docstring).
            score_func = f_classif if is_clf else f_regression
            self.feature_selector_ = SelectKBest(score_func=score_func, k=n_features)
        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            score_func = mutual_info_classif if is_clf else mutual_info_regression
            rs = self.random_state
            self.feature_selector_ = SelectKBest(
                score_func=lambda Xf, yf: score_func(Xf, yf, random_state=rs), k=n_features)
        else:  # "model_based"
            from sklearn.feature_selection import SelectFromModel
            base = (RandomForestClassifier(n_estimators=100, random_state=self.random_state,
                                           n_jobs=1)
                    if is_clf else
                    RandomForestRegressor(n_estimators=100, random_state=self.random_state,
                                          n_jobs=1))
            # threshold=-inf with max_features makes this a strict top-k by importance, so the
            # kept-feature count matches the other two methods and results stay comparable.
            self.feature_selector_ = SelectFromModel(base, max_features=n_features,
                                                     threshold=-np.inf)

        self.feature_selector_.fit(X, y)
        logger.info(f"Selected {n_features} of {X.shape[1]} features "
                    f"(method='{method}')")
    
    def _handle_class_imbalance(self, y: np.ndarray):
        """Handle class imbalance by computing class weights."""
        if self.task_type == 'classification' and self.handle_imbalanced:
            try:
                classes = np.unique(y)
                class_weights = compute_class_weight('balanced', classes=classes, y=y)
                self.class_weights_ = dict(zip(classes, class_weights))
                logger.info(f"Computed class weights: {self.class_weights_}")
            except Exception as e:
                logger.warning(f"Class weight computation failed: {str(e)}")
                self.class_weights_ = None
    
    def _compute_load_balance_loss(self, routing_weights: np.ndarray) -> float:
        """IMPROVEMENT 4: Compute load balancing loss."""
        try:
            expert_loads = routing_weights.sum(axis=0)
            if expert_loads.sum() > 0:
                mean_load = expert_loads.mean()
                if mean_load > 0:
                    loss = expert_loads.std() / (mean_load + 1e-8)
                    return loss
        except Exception:
            pass
        return 0.0
    
    def _apply_temperature_scaling(self, logits: np.ndarray) -> np.ndarray:
        """IMPROVEMENT 8: Apply temperature scaling to routing logits."""
        temp = max(0.1, self.routing_temperature)
        clipped = np.clip(logits, 1e-10, 1.0 - 1e-10)
        scaled = np.exp(np.log(clipped) / temp)
        normalized = scaled / scaled.sum(axis=1, keepdims=True)
        return normalized
    
    def _spawn_dynamic_track(self, X_uncertain: np.ndarray, y_uncertain: np.ndarray) -> bool:
        """IMPROVEMENT 9: Dynamically spawn new expert tracks."""
        if self._dynamic_tracks_created >= self.max_dynamic_tracks:
            return False
        if len(X_uncertain) < 20:
            return False
        
        try:
            track_name = f"track_dynamic_{self._dynamic_tracks_created}"
            models = self._create_heterogeneous_models(n_samples=len(X_uncertain))  # FIX 6: adapt n_estimators to this dataset size
            clf = models[self._dynamic_tracks_created % len(models)]
            
            feature_indices = None
            X_track_clf = X_uncertain
            _subset_mode = getattr(self, 'expert_feature_subsets', 'auto')  # PHASE1-F2
            _subsets_on = (bool(self.feature_selection) if _subset_mode == 'legacy'
                           else _subset_mode != 'off')
            if _subsets_on and X_uncertain.shape[1] > 3 and self.feature_dropout_rate != 1.0:
                # ACC-FIX 2: fixed keep-fraction when set, else preserve the exact original random range
                keep_frac = np.random.uniform(0.6, 0.8) if self.feature_dropout_rate is None else self.feature_dropout_rate
                n_select = max(2, int(X_uncertain.shape[1] * keep_frac))
                feature_indices = np.random.choice(X_uncertain.shape[1], size=n_select, replace=False)
                feature_indices.sort()
                X_track_clf = X_uncertain[:, feature_indices]
            
            self._fit_track_model(clf, X_track_clf, y_uncertain)
            track = Track(track_name, clf, feature_indices=feature_indices,
                         expert_capacity=self.expert_capacity)
            self.tracks[track_name] = track
            self._dynamic_tracks_created += 1
            logger.info(f"Spawned dynamic track '{track_name}' ({len(X_uncertain)} samples)")
            return True
        except Exception as e:
            logger.debug(f"Dynamic track spawning failed: {e}")
            return False
    
    def _create_tracks(self, X: np.ndarray, y: np.ndarray, bagging_mode_override: Optional[str] = None):
        """Create and train tracks with enhanced sampling or KMeans clustering."""
        logger.info(f"Creating {self.n_tracks} heterogeneous expert tracks...")
        
        # ACC-FIX 1: allows fit() to pass an effective bagging_mode for this call (e.g. when
        # small_data_auto_mode silently upgrades "bootstrap" to "full") without mutating the
        # self.bagging_mode hyperparameter itself.
        active_bagging_mode = bagging_mode_override if bagging_mode_override is not None else self.bagging_mode
        
        n_samples = X.shape[0]
        
        # IMPROVEMENT 10: KMeans Clustering for Track Specialization
        cluster_labels = None
        if self.cluster_experts and n_samples >= self.n_tracks:
            try:
                logger.info(f"Training KMeans for {self.n_tracks} expert regions...")
                self.kmeans_ = KMeans(n_clusters=self.n_tracks, random_state=self.random_state)
                cluster_labels = self.kmeans_.fit_predict(X)
                logger.info("KMeans clustering completed")
            except Exception as e:
                logger.warning(f"KMeans failed: {e}. Using bootstrap.")
                self.cluster_experts = False
        
        # IMPROVEMENT 2: Get heterogeneous models (FIX 6: adapt n_estimators to this dataset's size)
        models = self._create_heterogeneous_models(n_samples=n_samples)
        
        # IMPROVEMENT 3: Set expert capacity
        if self.expert_capacity is None:
            self.expert_capacity = n_samples / self.n_tracks
        
        # ACC2-FIX 4: running per-sample error accumulator used by diversity_reweighting.
        # Updated incrementally within Phase 1 (sequential) via a cheap probe fit of each
        # track, BEFORE the parallel Phase 2 below runs the real, final .fit() calls — this
        # keeps the row/feature/model-selection phase sequential (as it already is per FIX 3)
        # while preserving parallelism for the actual per-track training cost.
        cumulative_error_ = np.zeros(n_samples) if self.diversity_reweighting else None

        # FIX 3: Phase 1 — sequential data selection. All np.random calls happen here, in track
        # order, on the main thread, so results stay identical to the old sequential loop
        # regardless of how the subsequent .fit() calls are scheduled across threads.
        track_specs = []
        for i in range(self.n_tracks):
            track_name = f"track_{i}"
            
            # Data selection for this track
            if self.cluster_experts and cluster_labels is not None:
                cluster_indices = np.where(cluster_labels == i)[0]
                if len(cluster_indices) < 10:
                    logger.warning(f"Cluster {i} too small, using bootstrap")
                    indices = np.random.choice(n_samples, size=n_samples, replace=True)
                else:
                    other_indices = np.where(cluster_labels != i)[0]
                    if len(other_indices) > 0:
                        overlap_size = int(len(cluster_indices) * 0.2)
                        overlap = np.random.choice(other_indices, 
                                                 size=min(overlap_size, len(other_indices)), 
                                                 replace=False)
                        indices = np.concatenate([cluster_indices, overlap])
                    else:
                        indices = cluster_indices
                    np.random.shuffle(indices)
                
                X_track = X[indices]
                y_track = y[indices]
            else:
                if active_bagging_mode == "full":
                    # ACC-FIX 1: no row bootstrap — track sees the complete training set, relying
                    # only on differing model types / random_state offsets / hyperparameters (and
                    # feature dropout, if enabled) for diversity instead of row subsampling.
                    indices = np.arange(n_samples)
                    X_track = X
                    y_track = y
                else:
                    # Bootstrap sampling. PHASE1-F4: drawn from a per-track local Generator so
                    # the sample is reproducible from random_state alone and is genuinely
                    # different for each track, independent of whatever else has touched the
                    # global NumPy stream.
                    boot_rng = np.random.default_rng(
                        None if self.random_state is None else self.random_state + i * 100 + 13)
                    indices = boot_rng.choice(n_samples, size=n_samples, replace=True)
                    X_track = X[indices]
                    y_track = y[indices]
            
            # Feature selection
            feature_indices = None
            X_track_clf = X_track
            
            # PHASE1-F2: decouple per-expert feature subsetting from `feature_selection`.
            #
            # Feature DROPOUT is a diversity mechanism (give each expert a different view of the
            # data); feature SELECTION is a dimensionality mechanism (drop uninformative columns
            # globally). The original gate made the former unreachable unless the latter was on,
            # so any run with feature_selection=False trained every expert on identical columns
            # and identical rows-modulo-bootstrap — a direct cause of the 0.65-0.75 mean
            # pairwise error correlation measured by get_diversity_report().
            subset_mode = getattr(self, 'expert_feature_subsets', 'auto')
            if subset_mode == 'legacy':
                subsets_enabled = bool(self.feature_selection)
            elif subset_mode == 'off':
                subsets_enabled = False
            else:  # "auto"
                subsets_enabled = True

            if subsets_enabled and X.shape[1] > 3 and not self.cluster_experts and self.feature_dropout_rate != 1.0:
                # PHASE1-F4: local Generator instead of np.random.seed().
                #
                # The previous code seeded the GLOBAL stream to random_state + i*100, drew, then
                # reseeded it back to random_state. The next track's row bootstrap then drew from
                # that freshly-reset global stream, so every track after the first received an
                # IDENTICAL bootstrap sample — silently destroying the row diversity that
                # bagging_mode="bootstrap" exists to create. A local generator fixes both the
                # spec rule-16 violation and the diversity defect.
                feat_rng = np.random.default_rng(
                    None if self.random_state is None else self.random_state + i * 100 + 7)
                # ACC-FIX 2: fixed keep-fraction when set, else preserve the original random range
                keep_frac = (feat_rng.uniform(0.7, 0.9) if self.feature_dropout_rate is None
                             else self.feature_dropout_rate)
                n_select = max(2, int(X.shape[1] * keep_frac))
                feature_indices = feat_rng.choice(X.shape[1], size=n_select, replace=False)
                feature_indices.sort()
                X_track_clf = X_track[:, feature_indices]
            
            # FIX 3: clone the (possibly cycled/shared) model instance so each track gets its own
            # estimator object — required for safe concurrent .fit() calls below, since the
            # original list can contain repeated references when heterogeneous model types < n_tracks.
            clf = self._clone_track_model(models[i % len(models)])

            # ACC2-FIX 2: optionally wrap the classifier so its predict_proba is calibrated before
            # combination (routing OR stacking). SVM tracks are already wrapped in
            # CalibratedClassifierCV inside a Pipeline by FIX 1 for speed reasons, so we don't
            # double-wrap those — every other model type gets calibrated here.
            if self.calibrate_tracks and self.task_type == "classification":
                from sklearn.pipeline import Pipeline as _Pipeline
                already_calibrated = (
                    isinstance(clf, _Pipeline)
                    and isinstance(clf.named_steps.get('svm'), CalibratedClassifierCV)
                )
                if not already_calibrated:
                    clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
                    logger.debug(f"ACC2-FIX 2: track {track_name} classifier wrapped with "
                                 f"CalibratedClassifierCV (sigmoid, cv=3) for calibrated probabilities. "
                                 f"Per-track Brier-score-improvement logging is skipped here since a "
                                 f"proper held-out eval would require an extra per-track fit/split; "
                                 f"only the wrapping (which sklearn itself cross-validates internally) is applied.")

            # ACC2-FIX 4: diversity reweighting — for track i>0, upweight rows that the tracks
            # already probed in this loop (0..i-1) tended to get wrong, so this track is pushed to
            # specialize where the ensemble-so-far is weakest. Only applies to bootstrap/full
            # bagging (row-subset diversity is the mechanism being reinforced here).
            sample_weight = None
            if self.diversity_reweighting and active_bagging_mode in ("bootstrap", "full"):
                if i > 0:
                    row_errors = cumulative_error_[indices] / float(i)
                    # Bound to [1.0, 2.0] so a handful of persistently-hard samples can't blow up
                    # a track's effective loss function and destabilize training.
                    sample_weight = 1.0 + np.clip(row_errors, 0.0, 1.0)
                # CB-FIX 3: leak-free OOF probe fit (Phase 1) for the diversity signal. The
                # resulting error estimate is more expensive than the old single memorizing probe,
                # but each row is scored only on folds that did not train on it.
                try:
                    probe_errors = self._compute_diversity_probe_oof_errors(
                        clf, X_track_clf, y_track, sample_weight=sample_weight
                    )
                    np.add.at(cumulative_error_, indices, probe_errors)
                except Exception as e:
                    logger.debug(f"CB-FIX 3: diversity-reweighting OOF probe for track {i} failed: {e}")

            track_specs.append((track_name, clf, X_track_clf, y_track, feature_indices, sample_weight))
        
        # FIX 3: Phase 2 — parallelize the independent per-track .fit() calls, which is where
        # the actual training cost lives. ThreadPoolExecutor is safe here since each track now
        # fits its own cloned estimator on its own pre-selected data (no shared mutable state).
        def _fit_track(spec):
            name, clf, X_clf, y_clf, feat_idx, sample_weight = spec
            # ACC2-FIX 4: pass through the diversity-reweighting sample_weight computed in Phase 1,
            # when present and the underlying estimator actually supports it.
            if sample_weight is not None and has_fit_parameter(clf, 'sample_weight'):
                try:
                    self._fit_track_model(clf, X_clf, y_clf, sample_weight=sample_weight)
                    return name, clf, feat_idx
                except Exception as e:
                    logger.debug(f"ACC2-FIX 4: sample_weight fit failed for {name}, falling back to unweighted: {e}")
            self._fit_track_model(clf, X_clf, y_clf)
            return name, clf, feat_idx
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {executor.submit(_fit_track, spec): spec[0] for spec in track_specs}
            for future in as_completed(future_to_name):
                name, fitted_clf, feat_idx = future.result()
                results[name] = (fitted_clf, feat_idx)
        
        # Preserve original insertion order into self.tracks for reproducible downstream iteration
        for track_name, _clf, _X_clf, _y_clf, _feat_idx, _sw in track_specs:
            fitted_clf, feat_idx = results[track_name]
            track = Track(track_name, fitted_clf, feature_indices=feat_idx,
                         expert_capacity=self.expert_capacity)
            self.tracks[track_name] = track
        
        logger.info(f"Created {len(self.tracks)} heterogeneous tracks (clustering={self.cluster_experts})")
    
    def _align_proba(self, proba: np.ndarray, track_classes: np.ndarray) -> np.ndarray:
        """Align a track's probability output to the global class space.
        
        BUG FIX: When cluster-expert tracks train on a data subset, they may not
        see all global classes. Their predict_proba returns a (n, k) matrix where
        k < n_global_classes. This method maps the track's class columns back into
        the full (n, n_global_classes) probability matrix, with 0.0 for unseen classes.
        """
        n_global = len(self.classes_)
        if proba.shape[1] == n_global:
            return proba  # Already aligned, fast path
        
        aligned = np.zeros((proba.shape[0], n_global))
        for local_idx, cls in enumerate(track_classes):
            global_idx = np.where(self.classes_ == cls)[0]
            if len(global_idx) > 0:
                aligned[:, global_idx[0]] = proba[:, local_idx]
        return aligned

    def _apply_loss_aware_blend(self, X_selected: np.ndarray, router_weights: np.ndarray,
                                 track_names: List[str]) -> np.ndarray:
        """PHASE2 (spec section 3): blend the router's prior weights with each track's
        PREDICTED EXPECTED LOSS on this sample, using self.expert_loss_model_ (trained in
        _fit_loss_aware_router on leak-free OOF loss labels). This is the core "which expert
        minimizes expected loss" upgrade over pure "which expert is most likely correct"
        routing — see class-level docs / PHASE2 changelog for the full formulation.

            utility_e(x)  = -predicted_expected_loss_e(x)
            loss_weights   = softmax(utility / loss_aware_temperature)
            final          = (1 - loss_aware_blend) * router_prior + loss_aware_blend * loss_weights

        loss_aware_blend and loss_aware_temperature are explicit constructor parameters
        (never hardcoded), per spec section 3's "do not hard-code arbitrary weights" requirement.
        Falls back silently to the unmodified router_weights if the loss model can't be applied
        (e.g. track set changed since it was trained) — routing must never hard-fail because an
        optional refinement failed.
        """
        try:
            predicted_loss = np.atleast_2d(self.expert_loss_model_.predict(X_selected))
            if predicted_loss.shape[1] != len(track_names):
                aligned = np.full((predicted_loss.shape[0], len(track_names)), np.nanmean(predicted_loss))
                for i, name in enumerate(track_names):
                    if name in self._loss_aware_track_names_:
                        src = self._loss_aware_track_names_.index(name)
                        aligned[:, i] = predicted_loss[:, src]
                predicted_loss = aligned

            # PHASE1-F1: normalise ACROSS EXPERTS, per sample, before the softmax.
            #
            # The softmax is scale-sensitive but the loss is not scale-free: 1 - p_true lives on
            # [0,1] while |y - yhat| lives on the scale of the target. Feeding raw losses to
            # exp(-L/T) therefore produced a near-uniform blend for classification and a
            # saturated one-hot blend for regression, from the SAME code path and the same
            # default temperature. Standardising per row makes the softmax argument
            # dimensionless, so loss_aware_temperature means the same thing on every task.
            #
            # Rows where all experts have (near-)identical predicted loss have no signal to
            # extract; they normalise to all-zeros, which yields a uniform blend — the correct
            # behaviour, since the loss model is saying it cannot distinguish the experts here.
            mode = getattr(self, 'loss_aware_normalization', 'zscore')
            if mode == "zscore":
                centre = predicted_loss.mean(axis=1, keepdims=True)
                scale = predicted_loss.std(axis=1, keepdims=True)
                scale = np.where(scale < 1e-12, 1.0, scale)
                scaled_loss = (predicted_loss - centre) / scale
            elif mode == "range":
                lo = predicted_loss.min(axis=1, keepdims=True)
                hi = predicted_loss.max(axis=1, keepdims=True)
                span = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
                scaled_loss = (predicted_loss - lo) / span
            else:  # "none" — pre-PHASE1 behaviour, kept for reproducibility of old runs
                scaled_loss = predicted_loss

            temp = max(1e-3, self.loss_aware_temperature)
            utility = -scaled_loss / temp
            utility = utility - utility.max(axis=1, keepdims=True)  # numerical stability
            loss_weights = np.exp(utility)
            loss_weights /= loss_weights.sum(axis=1, keepdims=True)

            blend = float(np.clip(self.loss_aware_blend, 0.0, 1.0))
            blended = (1 - blend) * router_weights + blend * loss_weights
            row_sums = blended.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return blended / row_sums
        except Exception as e:
            logger.debug(f"PHASE2: loss-aware blend failed, falling back to router-only weights: {e}")
            return router_weights

    def _route_with_router(self, X_selected: np.ndarray, track_output_cache: Optional[dict] = None) -> Dict[str, Any]:
        """Shared MoE routing decision for the (non-intelligent, multi-track) router path.

        predict() and predict_proba() previously each reimplemented this independently, which
        both duplicated ~15 lines of temperature-sharpening/expansion logic (spec item 32) and
        created a real risk of the two entry points silently diverging (spec item N: "predict/
        predict_proba should share a single coherent routing/fusion path"). Both now call this.

        IMPORTANT backward-compatibility note: when self.router_loss_aware is False (the
        default), best_track_indices/routing_confidences are computed from the RAW router_probas
        exactly as before PHASE2 — hard-routing choice and abstention/fallback-threshold
        semantics are byte-for-byte unchanged for existing users who haven't opted into
        loss-aware routing. Only when router_loss_aware=True do these reflect the loss-aware-
        blended weights instead (an explicit, documented semantic change gated behind that flag).
        """
        n_samples = len(X_selected)
        X_for_router = X_selected
        if getattr(self, '_router_uses_meta_features', False):
            X_for_router = self._build_router_meta_features(X_selected, track_output_cache=track_output_cache)
        router_probas = self.router_.predict_proba(X_for_router)

        n_tracks_total = len(self.router_track_names_)
        router_classes = getattr(self.router_, 'classes_', np.arange(n_tracks_total))
        temp = max(0.1, self.routing_temperature)
        sharpened_raw = np.exp(np.log(np.clip(router_probas, 1e-9, 1.0)) / temp)
        sharpened_raw /= sharpened_raw.sum(axis=1, keepdims=True)
        router_weights = np.zeros((n_samples, n_tracks_total))
        for local_i, global_track_idx in enumerate(router_classes):
            if global_track_idx < n_tracks_total:
                router_weights[:, global_track_idx] = sharpened_raw[:, local_i]
        row_sums = router_weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        router_weights /= row_sums

        if self.router_loss_aware and getattr(self, 'expert_loss_model_', None) is not None:
            router_weights = self._apply_loss_aware_blend(X_selected, router_weights, self.router_track_names_)
            best_track_indices = np.argmax(router_weights, axis=1)
            routing_confidences = np.max(router_weights, axis=1)
        else:
            # Exact pre-PHASE2 behavior: index/confidence come from the router's own
            # (unsharpened, unexpanded) probability output.
            # IMPROVEMENT: Router columns are local positions, not necessarily track IDs.
            best_track_indices = np.asarray(router_classes, dtype=int)[np.argmax(router_probas, axis=1)]
            routing_confidences = np.max(router_probas, axis=1)
        current_tracks = np.array([self.router_track_names_[i] for i in best_track_indices])

        return {
            'router_probas': router_probas,
            'router_weights': router_weights,
            'best_track_indices': best_track_indices,
            'routing_confidences': routing_confidences,
            'current_tracks': current_tracks,
        }

    def _build_router_meta_features(self, X: np.ndarray, track_output_cache: Optional[dict] = None) -> np.ndarray:
        """
        IMPROVEMENT 7 + SIGNAL-GUIDED ROUTING: Build meta-features for the router.
        
        Augments input X with:
        1. Cross-track disagreement signals (IMPROVEMENT 7)
        2. Structural signals: expert disagreement, entropy, density, cluster distance, outlier score
        
        This makes the router aware of both track consensus AND data structure.
        
        FIX 4: Accepts an optional `track_output_cache` dict (track_name -> raw
        predict/predict_proba output on this exact X). If a caller already computed a
        track's output for this X (e.g. during router training, or the final aggregation
        step in predict()/predict_proba()), passing that cache here avoids recomputing it.
        Any track not yet in the cache gets computed once and stored back into it, so the
        cache can be reused again by the caller afterwards.
        """
        if not self.use_meta_features or len(self.tracks) < 2:
            return X
        
        try:
            track_names = [t for t in self.tracks.keys() if t != 'correction_track']
            meta_cols = []
            track_predictions = []
            
            # Collect track predictions
            for track_name in track_names:
                if track_name not in self.tracks:
                    continue
                track = self.tracks[track_name]
                
                try:
                    if self.task_type == 'classification':
                        if track_output_cache is not None and track_name in track_output_cache:
                            proba = track_output_cache[track_name]  # FIX 4: reuse cached output
                        else:
                            proba = track.predict_proba(X)
                            if track_output_cache is not None:
                                track_output_cache[track_name] = proba  # FIX 4: cache for reuse by caller
                        if proba.shape[1] > 0:
                            meta_cols.append(proba[:, 0].reshape(-1, 1))
                            track_predictions.append(proba[:, 0])
                    else:
                        if track_output_cache is not None and track_name in track_output_cache:
                            preds = track_output_cache[track_name]  # FIX 4: reuse cached output
                        else:
                            preds = track.predict(X).astype(float)
                            if track_output_cache is not None:
                                track_output_cache[track_name] = preds  # FIX 4: cache for reuse by caller
                        meta_cols.append(preds.reshape(-1, 1))
                        track_predictions.append(preds)
                except Exception:
                    meta_cols.append(np.zeros((len(X), 1)))
                    track_predictions.append(np.zeros(len(X)))
            
            # Add track consensus features
            X_augmented = X.copy() if isinstance(X, np.ndarray) else X
            
            if len(meta_cols) >= 2:
                track_matrix = np.hstack(meta_cols)
                disagreement = track_matrix.std(axis=1, keepdims=True)
                disagreement = np.nan_to_num(disagreement, 0.0)
                X_augmented = np.hstack([X, track_matrix, disagreement])
            
            # SIGNAL-GUIDED ROUTING: Add structural signals
            if self.signal_extractor_ is not None and self.signal_extractor_.fitted_:
                try:
                    # Prepare track predictions for signal extraction
                    track_preds_array = np.column_stack(track_predictions) if track_predictions else None

                    # PHASE7 FIX: mean-ensemble probability (classification only), built from
                    # the same per-track outputs already cached above -- see extract_signals'
                    # docstring for why a router probability was never actually available here.
                    ensemble_proba = None
                    if self.task_type == 'classification' and track_output_cache:
                        aligned_probas = []
                        for tname in track_names:
                            raw = track_output_cache.get(tname)
                            if raw is None or np.asarray(raw).ndim != 2:
                                continue
                            track_classes = getattr(self.tracks[tname].classifier, 'classes_', self.classes_)
                            aligned_probas.append(self._align_proba(raw, track_classes))
                        if aligned_probas:
                            ensemble_proba = np.mean(aligned_probas, axis=0)

                    # Extract structural signals
                    signals = self.signal_extractor_.extract_signals(
                        X, track_preds_array, ensemble_proba=ensemble_proba)
                    
                    # Augment with structural signals
                    X_augmented = np.hstack([X_augmented, signals])
                    logger.debug(f"Added {signals.shape[1]} structural signals for routing guidance")
                except Exception as e:
                    logger.debug(f"Structural signal extraction failed: {e}")
            
            if X_augmented.shape[1] > X.shape[1]:
                logger.debug(f"Router meta-features: {X.shape[1]} → {X_augmented.shape[1]}")
                return X_augmented
                
        except Exception as e:
            logger.debug(f"Meta-feature augmentation failed: {e}")
        
        return X
    
    def _build_router_features(self, X: np.ndarray) -> np.ndarray:
        """DEPRECATED: Use _build_router_meta_features instead."""
        return self._build_router_meta_features(X)

    def _compute_abstain_mask(self, X_selected: np.ndarray, routing_confidences: np.ndarray) -> np.ndarray:
        """PHASE5 (spec item 26): resolve the abstention decision mask.

        Default (abstention_use_difficulty=False): EXACT original behavior — abstain where
        routing_confidences < abstention_threshold. Kept as the default so existing users see
        no behavior change.

        abstention_use_difficulty=True: abstain where predict_difficulty(X) > abstention_threshold
        instead — a genuine expected-risk criterion (trained against real OOF ensemble loss,
        see _fit_difficulty_model) rather than the router's confidence in which track it picked.
        This matters most for regression, where routing_confidences was never a measure of
        prediction reliability at all, and even for classification it directly implements spec
        item 26's "abstain if expected risk > threshold" framing. Falls back to the original
        routing_confidences behavior (with a logged warning) if no difficulty model is
        available for this fit.
        """
        if self.abstention_use_difficulty:
            if self.difficulty_model_ is not None:
                try:
                    difficulty = self._difficulty_from_selected(X_selected)
                    return difficulty > self.abstention_threshold
                except Exception as e:
                    logger.warning(f"PHASE5: difficulty-based abstention failed ({e}); "
                                    f"falling back to routing-confidence-based abstention.")
            else:
                logger.warning("PHASE5: abstention_use_difficulty=True but no difficulty model "
                                "is available for this fit (enable_difficulty_model=False, or "
                                "too little data); falling back to routing-confidence-based "
                                "abstention.")
        return routing_confidences < self.abstention_threshold

    def _get_abstention_sentinel(self) -> Any:
        """PHASE1-FIX: resolve an unambiguous abstention sentinel value.

        The original default (bare -1) silently collides with real class labels once labels
        are non-integer or non-contiguous — e.g. with string labels, numpy coerces -1 into the
        string '"-1"' on assignment into an object/string prediction array instead of raising,
        so an abstained sample becomes indistinguishable from a genuine (if unlikely) '"-1"'
        class label. This picks a sentinel that provably cannot collide with self.classes_:
        the user's explicit abstention_class if given, else -1 only when classes_ is integer
        and -1 isn't already an in-use label, else a distinct string marker.
        """
        if self.abstention_class is not None:
            return self.abstention_class
        if self.task_type == "classification" and getattr(self, 'classes_', None) is not None:
            if np.issubdtype(self.classes_.dtype, np.integer) and -1 not in self.classes_:
                return -1
            return "__TRA_ABSTAIN__"
        return -1

    def _assign_with_dtype_promotion(self, arr: np.ndarray, mask: np.ndarray, value: Any) -> np.ndarray:
        """PHASE1-FIX: assign `value` into `arr[mask]`, promoting `arr` to object dtype first
        whenever `value` cannot be represented in arr's current dtype without silent, lossy
        coercion (e.g. an int sentinel into a fixed-width unicode array, or vice versa)."""
        if arr.dtype != object:
            try:
                round_tripped = np.array([value], dtype=arr.dtype)[0]
                # Reject silent coercions where the value doesn't survive the round trip
                # (e.g. int -1 -> numpy string '-1' compares unequal to the original int).
                if not (round_tripped == value):
                    raise ValueError("lossy dtype coercion")
            except (ValueError, TypeError):
                arr = arr.astype(object)
        arr[mask] = value
        return arr

    def _add_correction_track(self, X_train: np.ndarray, y_train: np.ndarray,
                               oof_out: Optional[np.ndarray] = None):
        """IMPROVEMENT 3 / PHASE1-FIX D: Add a Residual Correction Track (TRA-Boost).

        After all expert tracks are trained, a final 'correction track' is trained to
        correct systematic errors the routing layer cannot fix alone. This is analogous
        to one boosting round on top of the MoE ensemble.

        PHASE1-FIX D (leakage): the ORIGINAL implementation computed "ensemble predictions"
        by calling track.predict(X_train) on tracks that were themselves trained on (all or
        an overlapping bootstrap of) X_train — i.e. in-sample predictions. Those predictions
        are overly optimistic (tracks partially "remember" their own training rows), so the
        correction track was trained on an artificially easy/biased failure set. This version
        uses genuinely out-of-fold (OOF) ensemble predictions instead — each sample's ensemble
        prediction comes from track clones that never saw that sample — matching the same
        leak-free OOF machinery already used for stacking/router training (ACC-FIX 4/5).

        Gating (spec item 14): the correction track is only kept if it demonstrates a real,
        held-out improvement. We further split the OOF-identified failure region into a
        fit/gate split so the reported gain is itself not measured on the same rows the final
        correction model is fit on. If the gain does not clear self.correction_gain_threshold,
        the correction track is discarded (self.correction_track_ stays None) rather than
        deployed on faith.
        """
        try:
            logger.info("Training residual correction track (TRA-Boost) on OOF ensemble errors...")
            n_samples = len(X_train)

            if oof_out is None:
                oof_out = self._compute_oof_track_outputs(X_train, y_train, self.stacking_folds)

            if self.task_type == 'classification':
                # Ensemble OOF probability = mean of the (already class-aligned) per-slot OOF
                # probabilities; this is a leak-free stand-in for "what would the full ensemble
                # have predicted on this row if it hadn't seen it during training".
                ensemble_oof_proba = oof_out.mean(axis=1)  # (n_samples, n_classes)
                ensemble_oof_preds = self.classes_[np.argmax(ensemble_oof_proba, axis=1)]
                wrong_mask = (ensemble_oof_preds != y_train)
                n_wrong = int(wrong_mask.sum())

                if n_wrong <= 20:
                    logger.info(f"Too few OOF ensemble errors ({n_wrong}) for correction track, skipping.")
                    self.correction_track_ = None
                    return

                X_wrong, y_wrong = X_train[wrong_mask], y_train[wrong_mask]

                # PHASE3-G1: validate the DEPLOYED rule, not a proxy for it.
                #
                # The previous gate fitted a probe on part of the OOF-wrong rows and scored it
                # on the rest, calling the resulting accuracy "gain" against an implicit 0%
                # baseline. That measures something real but not the thing that matters: in
                # production the correction model is applied to rows chosen by a confidence
                # rule, most of which the ensemble got RIGHT, and overriding those is a cost the
                # 0%-baseline framing cannot represent. Gains of 0.62-0.75 were reported by that
                # gate while the measured deployment application rate was ~0.000.
                #
                # Here a probe is fitted on the OOF-wrong rows WITHIN a fit split, and then the
                # actual override rule is replayed on a held-out split at several risk
                # quantiles, scored as full accuracy against the uncorrected OOF ensemble. A
                # threshold is adopted only if it beats doing nothing by
                # correction_gain_threshold.
                risk_all = 1.0 - np.max(ensemble_oof_proba, axis=1)
                try:
                    fit_rows, gate_rows = train_test_split(
                        np.arange(n_samples), test_size=0.3, random_state=self.random_state,
                        stratify=y_train if len(np.unique(y_train)) > 1 else None)
                except ValueError:
                    fit_rows, gate_rows = np.arange(n_samples), np.array([], dtype=int)

                best_gain, best_threshold = 0.0, None
                if len(gate_rows) > 0:
                    fit_wrong = fit_rows[wrong_mask[fit_rows]]
                    if len(fit_wrong) >= 10 and len(np.unique(y_train[fit_wrong])) > 1:
                        probe_clf = self._get_base_estimator()
                        try:
                            self._fit_track_model(probe_clf, X_train[fit_wrong], y_train[fit_wrong])
                            gate_y = y_train[gate_rows]
                            base_pred = ensemble_oof_preds[gate_rows]
                            base_acc = float(np.mean(base_pred == gate_y))
                            probe_pred = probe_clf.predict(X_train[gate_rows])
                            try:
                                probe_conf = np.max(probe_clf.predict_proba(X_train[gate_rows]), axis=1)
                            except Exception:
                                probe_conf = np.ones(len(gate_rows))
                            gate_risk = risk_all[gate_rows]
                            for q in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
                                thr = float(np.quantile(gate_risk, 1.0 - q))
                                apply_mask = ((gate_risk >= thr)
                                              & (probe_conf >= self.correction_confidence_threshold))
                                if not np.any(apply_mask):
                                    continue
                                candidate = base_pred.copy()
                                candidate[apply_mask] = probe_pred[apply_mask]
                                cand_gain = float(np.mean(candidate == gate_y)) - base_acc
                                if cand_gain > best_gain:
                                    best_gain, best_threshold = cand_gain, thr
                        except Exception as e:
                            logger.debug(f"Correction-track gating probe failed: {e}")

                self.correction_gain_ = best_gain
                if best_threshold is None or best_gain < self.correction_gain_threshold:
                    logger.info(f"Correction track best held-out accuracy gain ({best_gain:.4f}) is "
                                f"below correction_gain_threshold ({self.correction_gain_threshold}) "
                                f"at every candidate operating point; discarding correction track "
                                f"rather than shipping an inactive one.")
                    self.correction_track_ = None
                    self.correction_risk_threshold_ = None
                    return

                # Gate passed — refit on the FULL OOF-wrong set for deployment and keep the
                # validated operating point.
                clf = self._get_base_estimator()
                self._fit_track_model(clf, X_wrong, y_wrong)
                self.correction_track_ = Track('correction_track', clf)
                self.correction_risk_threshold_ = best_threshold
                logger.info(f"Correction track trained on {n_wrong} OOF-misclassified samples "
                            f"(validated held-out accuracy gain={best_gain:+.4f} at risk "
                            f"threshold {best_threshold:.4f}).")
            else:
                # Regression: residuals against the leak-free OOF ensemble prediction.
                ensemble_oof_preds = oof_out.mean(axis=1)  # (n_samples,)
                residuals = y_train - ensemble_oof_preds
                baseline_mae = float(np.mean(np.abs(residuals)))
                if baseline_mae <= 1e-8:
                    logger.info("OOF ensemble residuals are ~0, skipping correction track.")
                    self.correction_track_ = None
                    return

                try:
                    fit_idx, gate_idx = train_test_split(
                        np.arange(n_samples), test_size=0.3, random_state=self.random_state
                    )
                except ValueError:
                    fit_idx, gate_idx = np.arange(n_samples), np.array([], dtype=int)

                gain = 0.0
                if len(gate_idx) > 0:
                    probe_clf = self._get_base_estimator()
                    try:
                        self._fit_track_model(probe_clf, X_train[fit_idx], residuals[fit_idx])
                        gate_correction = probe_clf.predict(X_train[gate_idx])
                        corrected_mae = float(np.mean(np.abs(residuals[gate_idx] - gate_correction)))
                        gate_baseline_mae = float(np.mean(np.abs(residuals[gate_idx])))
                        gain = (gate_baseline_mae - corrected_mae) / max(gate_baseline_mae, 1e-8)
                    except Exception as e:
                        logger.debug(f"Correction-track gating probe failed: {e}")
                        gain = 0.0
                else:
                    gain = self.correction_gain_threshold

                self.correction_gain_ = gain
                if gain < self.correction_gain_threshold:
                    logger.info(f"Correction track MAE-reduction gain ({gain:.4f}) below "
                                f"correction_gain_threshold ({self.correction_gain_threshold}); "
                                f"discarding correction track.")
                    self.correction_track_ = None
                    return

                clf = self._get_base_estimator()
                self._fit_track_model(clf, X_train, residuals)
                self.correction_track_ = Track('correction_track', clf)
                logger.info(f"Correction track trained on OOF residuals (mean |residual|={baseline_mae:.4f}, "
                            f"held-out MAE-reduction gain={gain:.4f})")
        except Exception as e:
            logger.warning(f"Correction track training failed: {e}")
            self.correction_track_ = None

    def _compute_oof_track_outputs(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """ACC-FIX 4 / ACC-FIX 5: shared out-of-fold (OOF) evaluation helper.

        Trains a FRESH clone of each CURRENT heterogeneous "track slot" (same model-type
        rotation _create_tracks uses) on k-1 folds and predicts on the held-out fold, so the
        returned per-slot outputs are genuinely unbiased — each sample's output comes from a
        model that never saw that sample during training. This is what makes it safe to use
        these outputs as either (a) meta-learner training features for stacking, (b) router
        training targets, or (c) PHASE3/4 diagnostics/refresh inputs, without the leakage that
        would come from scoring a track on rows it was itself bootstrap-sampled/trained from.

        PHASE4-FIX: n_slots is derived from the CURRENT len(self.tracks) (excluding the
        correction track) rather than the fixed self.n_tracks constructor hyperparameter.
        During an ordinary fit() call these are identical (tracks are freshly created to match
        n_tracks exactly, before this is ever invoked) — but they diverge once the track pool
        has been pruned or dynamically grown (spec item 12's refresh_fusion()/refresh_experts()
        call this function again against the CURRENT track set), and using the stale fixed
        n_tracks there produced a shape mismatch against the model's actual track count.

        Returns:
          classification: array of shape (n_samples, n_slots, n_classes) — per-slot aligned probabilities.
          regression: array of shape (n_samples, n_slots) — per-slot point predictions.
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        
        n_samples = len(X)
        # IMPROVEMENT: Preserve current expert identities and calibration wrappers in OOF clones.
        models_template = ([track.classifier for name, track in self.tracks.items()
                            if name != 'correction_track']
                           or self._create_heterogeneous_models(n_samples=n_samples))
        n_slots = max(1, len([t for t in self.tracks.keys() if t != 'correction_track'])) if self.tracks else self.n_tracks
        
        if self.task_type == "classification":
            n_classes = len(self.classes_)
            oof_out = np.zeros((n_samples, n_slots, n_classes))
            # IMPROVEMENT: Every stratified fold needs genuine support for every task class.
            min_class_count = int(np.unique(y, return_counts=True)[1].min())
            n_splits = min(k, max(2, n_samples // max(1, n_slots)), min_class_count)
            if n_splits < 2:
                raise ValueError("OOF classification requires at least two samples per class")
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(X, y)
        else:
            oof_out = np.zeros((n_samples, n_slots))
            splitter = KFold(n_splits=min(k, max(2, n_samples // max(1, n_slots))), shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(X)
        
        # ACC2-FIX 5: parallelize the per-fold, per-slot OOF fits with ThreadPoolExecutor, the same
        # way _create_tracks parallelizes track fitting. Each (fold, slot) combination fits its own
        # cloned estimator on its own fold split independently of every other combination, so this
        # is safe to run concurrently and cuts the stacking-mode training-time overhead observed in
        # benchmarking (e.g. ~12s vs ~5s on a ~1800-row/5-track case). Respects self.max_workers.
        fold_splits = list(split_iter)
        # IMPROVEMENT: Share read-only fold slices across experts instead of copying each slot.
        fold_data = [(X[tr], X[va], y[tr]) for tr, va in fold_splits]

        # PHASE1-F5: the OOF surrogate for slot `slot` must be the SAME KIND OF MODEL, ON THE
        # SAME FEATURE SUBSET, as the deployed track occupying that slot. The original
        # implementation cloned from models_template and fitted on the FULL feature matrix,
        # ignoring each track's `feature_indices`. Whenever per-track feature subsetting was
        # active, every OOF-derived quantity — router best-track labels, loss-aware expected-loss
        # targets, the difficulty model, correction-track residuals, stacking meta-features and
        # the diversity / marginal-contribution diagnostics — therefore described a set of
        # experts that DOES NOT EXIST at inference time. This is not label leakage (the folds are
        # honest), but it is an OOF-fidelity failure with the same practical consequence: the
        # router is trained to dispatch to models other than the ones it dispatches to.
        #
        # Diagnosed by observing that get_diversity_report() returned bit-identical error
        # correlations for two configurations whose deployed experts differed enough to move MAE
        # from 11.6 to 16.9 — the diagnostic was measuring the surrogates, not the experts.
        slot_track_names = [t for t in self.tracks.keys() if t != 'correction_track']

        def _slot_feature_indices(slot):
            if slot < len(slot_track_names):
                return getattr(self.tracks[slot_track_names[slot]], 'feature_indices', None)
            return None

        def _fit_oof_slot(fold_idx, train_idx, val_idx, slot):
            try:
                X_fold_train, X_fold_val, y_fold_train = fold_data[fold_idx]
                slot_features = _slot_feature_indices(slot)
                if slot_features is not None:
                    X_fold_train = X_fold_train[:, slot_features]
                    X_fold_val = X_fold_val[:, slot_features]
                fold_clf = self._clone_track_model(models_template[slot % len(models_template)])
                self._fit_track_model(fold_clf, X_fold_train, y_fold_train)
                if self.task_type == "classification":
                    proba = fold_clf.predict_proba(X_fold_val)
                    classes_local = getattr(fold_clf, 'classes_', self.classes_)
                    result = self._align_proba(proba, classes_local)
                else:
                    result = fold_clf.predict(X_fold_val)
                return fold_idx, slot, val_idx, result, None
            except Exception as e:
                return fold_idx, slot, val_idx, None, str(e)

        tasks = [
            (fold_idx, train_idx, val_idx, slot)
            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits)
            for slot in range(n_slots)
        ]

        oof_errors = []  # IMPROVEMENT: Failed OOF cells must never masquerade as zero predictions.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_fit_oof_slot, *task) for task in tasks]
            for future in as_completed(futures):
                fold_idx, slot, val_idx, result, err = future.result()
                if err is not None:
                    logger.warning(f"OOF fold {fold_idx} slot {slot} training/prediction failed: {err}")
                    oof_errors.append(f"fold {fold_idx}, track {slot}: {err}")
                    continue
                if self.task_type == "classification":
                    oof_out[val_idx, slot, :] = result
                else:
                    oof_out[val_idx, slot] = result

        # IMPROVEMENT: Stop fusion training when its OOF evidence is incomplete or invalid.
        if oof_errors:
            raise RuntimeError("Incomplete OOF predictions: " + "; ".join(oof_errors))
        if not np.isfinite(oof_out).all():
            raise ValueError("OOF predictions contain nonfinite values")
        if self.task_type == "classification" and (
                np.any(oof_out < 0) or not np.allclose(oof_out.sum(axis=2), 1.0)):
            raise ValueError("OOF class probabilities must be nonnegative and sum to one")
        return oof_out

    def _select_combination_mode_auto(self, X_selected: np.ndarray, y: np.ndarray) -> str:
        """ACC2-FIX 1: internal validation-based selector for combination_mode="auto".

        Runs a lightweight k-fold comparison (self.auto_select_folds folds, independent of
        self.stacking_folds so this stays fast even when stacking_folds is large for the real
        fit) that clones THIS estimator, forces combination_mode to "routing" and then
        "stacking" on each fold's training split, fits both clones, and scores both on the
        held-out fold with the task's native metric (accuracy for classification, negative RMSE
        for regression). Because this reuses clone(self).fit()/.predict() rather than a bespoke
        approximation, the comparison is a faithful preview of what each mode would actually do
        on this dataset. Whichever mode wins on average is used for the FINAL fit on the full
        training data back in fit().

        NOTE: X_selected here is already preprocessed/feature-selected (the same array fit()
        hands to _create_tracks), so each probe's own internal preprocessing step re-processes
        already-transformed features. That's a minor redundancy but keeps this selector cheap
        and self-contained; it does not change the columns being modeled by more than one
        additional standardization/selection pass.
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        n_samples = len(X_selected)
        n_folds = min(self.auto_select_folds, max(2, n_samples // max(4, self.n_tracks)))
        if n_folds < 2:
            logger.info("ACC2-FIX 1: not enough samples for an auto combination_mode comparison; "
                        "defaulting to 'routing'.")
            return "routing"

        if self.task_type == "classification":
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(X_selected, y)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            split_iter = splitter.split(X_selected)

        # PHASE3-G2: three candidates, not two.
        candidate_scores: Dict[str, List[float]] = {
            "routing": [], "stacking": [], "flat_average": []}

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            X_tr, X_val = X_selected[train_idx], X_selected[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            for mode, scores in candidate_scores.items():
                try:
                    probe = clone(self)
                    probe.combination_mode = mode
                    # Isolate this comparison from the legacy small-data heuristic and keep the
                    # probe's own internal stacking OOF pass cheap regardless of the outer
                    # stacking_folds setting.
                    probe.small_data_auto_mode = False
                    probe.stacking_folds = max(2, min(self.auto_select_folds, self.stacking_folds))
                    probe.fit(X_tr, y_tr)
                    y_pred = probe.predict(X_val)
                    if self.task_type == "classification":
                        score = accuracy_score(y_val, y_pred)
                    else:
                        score = -np.sqrt(mean_squared_error(y_val, y_pred))
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"ACC2-FIX 1: auto mode-selection fold {fold_idx} '{mode}' probe failed: {e}")

        means = {mode: (float(np.mean(s)) if s else float('-inf'))
                 for mode, s in candidate_scores.items()}
        self.combination_mode_scores_ = means  # exposed via get_diagnostics()
        # Ties resolve toward the cheaper mechanism: flat_average (no router, no meta-learner)
        # beats routing, which beats stacking. Preferring the simpler model when validation
        # cannot separate them is the whole point of running the comparison.
        preference = {"flat_average": 2, "routing": 1, "stacking": 0}
        chosen = max(means, key=lambda m: (means[m], preference[m]))
        summary = ", ".join(f"{m}={means[m]:.4f} (n={len(candidate_scores[m])})"
                            for m in ("routing", "stacking", "flat_average"))
        logger.info(f"ACC2-FIX 1 / PHASE3-G2: auto combination_mode comparison over "
                    f"{n_folds} folds — {summary} -> chosen '{chosen}'")
        return chosen

    def _fit_stacking(self, X_train: np.ndarray, y_train: np.ndarray, oof_out: Optional[np.ndarray] = None):
        """ACC-FIX 4: Train a stacking meta-learner on out-of-fold per-track outputs.

        Replaces router-based hard/soft track selection with a meta-learner
        (LogisticRegression for classification, Ridge for regression) fit on the stacked
        OOF outputs of every production track (self.tracks) across self.stacking_folds folds.
        Using OOF outputs (rather than scoring tracks on their own training rows) avoids the
        leakage that would otherwise make the meta-learner overconfident in tracks that simply
        memorized their bootstrap sample.

        PHASE7: when self.stacking_use_signals is True (default), the leak-free structural
        signals (expert disagreement, prediction entropy, feature density, cluster distance,
        outlier score) computed from these same OOF track outputs are concatenated onto the
        meta-feature matrix before fitting. This replaces the previous behavior, where stacking
        never consumed the signal extractor at all despite being the combination_mode this
        project's own benchmark recommends -- meaning "signal-guided" contributed nothing to
        TRA's best-performing, best-calibrated configuration. Set stacking_use_signals=False to
        reproduce the exact pre-PHASE7 stacking meta-learner.
        """
        track_names = [t for t in self.tracks.keys() if t != 'correction_track']
        n_tracks = len(track_names)
        n_samples = len(X_train)
        self._stacking_signals_active_ = False

        if n_tracks < 2:
            logger.warning("ACC-FIX 4: fewer than 2 tracks available for stacking; skipping meta-learner training.")
            self.meta_learner_ = None
            self._stacking_track_names_ = track_names
            return
        
        if oof_out is None:
            logger.info(f"ACC-FIX 4: Generating {self.stacking_folds}-fold OOF outputs for {n_tracks} tracks (stacking mode)...")
            oof_out = self._compute_oof_track_outputs(X_train, y_train, self.stacking_folds)
        else:
            logger.info(f"ACC-FIX 4 / PHASE1: reusing shared {self.stacking_folds}-fold OOF outputs for {n_tracks} tracks (stacking mode).")
        
        if self.task_type == "classification":
            n_classes = len(self.classes_)
            oof_matrix = oof_out.reshape(n_samples, n_tracks * n_classes)
        else:
            oof_matrix = oof_out.reshape(n_samples, n_tracks)

        # PHASE7: widen the meta-feature matrix with the 5 structural signals, computed from
        # the SAME OOF track outputs already in hand (no extra track fits needed). Recorded in
        # self._stacking_signals_active_ so _stacking_predict_input can guarantee it always
        # appends the same number of columns the meta-learner was actually fit on, even if
        # signal computation succeeds here but fails at inference time for some batch.
        self._stacking_signals_active_ = False
        if self.stacking_use_signals and self.signal_extractor_ is not None and self.signal_extractor_.fitted_:
            try:
                if self.task_type == "classification":
                    track_preds_for_signals = np.argmax(oof_out, axis=2).astype(float)
                    ensemble_proba = oof_out.mean(axis=1)
                else:
                    track_preds_for_signals = oof_out
                    ensemble_proba = None
                stacking_signals = self.signal_extractor_.extract_signals(
                    X_train, track_predictions=track_preds_for_signals, ensemble_proba=ensemble_proba)
                oof_matrix = np.hstack([oof_matrix, stacking_signals])
                self._stacking_signals_active_ = True
            except Exception as e:
                logger.debug(f"PHASE7: stacking signal augmentation failed, proceeding without it: {e}")

        try:
            if self.meta_learner_cv:
                # ACC2-FIX 3: cross-validated meta-learner with a small default regularization grid,
                # instead of the fixed-regularization LogisticRegression/Ridge below.
                #
                # PHASE7 FIX: scoring='neg_log_loss' is REQUIRED here, not cosmetic.
                # LogisticRegressionCV's default scoring is accuracy, which on separable/easy
                # datasets (measured directly on Wine) selects the LEAST regularized C in the
                # grid -- accuracy goes up (0.9717 -> 0.9830) while log loss goes up roughly
                # 7-FOLD (0.0685 -> 0.4965) and ECE roughly 10-fold, because the accuracy-
                # selected C produces overconfident probabilities that are catastrophically
                # penalized on the folds it gets wrong. Since calibration is this project's
                # measured, defensible advantage over plain stacking, tuning the meta-learner's
                # regularization to maximize accuracy while ignoring calibration was actively
                # working against the whole point of stacking_use_signals/meta_learner_cv
                # existing at all. Explicitly scoring on log loss selects regularization that
                # keeps probabilities honest instead.
                from sklearn.linear_model import LogisticRegressionCV, RidgeCV
                if self.task_type == "classification":
                    self.meta_learner_ = LogisticRegressionCV(
                        Cs=[0.01, 0.1, 1.0, 10.0, 100.0], cv=min(5, max(2, n_tracks)),
                        scoring="neg_log_loss", max_iter=1000, random_state=self.random_state
                    )
                else:
                    self.meta_learner_ = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                self.meta_learner_.fit(oof_matrix, y_train)
                self._stacking_track_names_ = track_names
                logger.info(f"ACC2-FIX 3: Stacking meta-learner (CV-regularized) trained on "
                            f"{oof_matrix.shape[1]} OOF meta-features from {n_tracks} tracks across "
                            f"{self.stacking_folds} folds.")
            else:
                from sklearn.linear_model import LogisticRegression, Ridge
                if self.task_type == "classification":
                    self.meta_learner_ = LogisticRegression(max_iter=1000, random_state=self.random_state)
                else:
                    self.meta_learner_ = Ridge(random_state=self.random_state)
                self.meta_learner_.fit(oof_matrix, y_train)
                self._stacking_track_names_ = track_names
                logger.info(f"ACC-FIX 4: Stacking meta-learner trained on {oof_matrix.shape[1]} OOF meta-features "
                            f"from {n_tracks} tracks across {self.stacking_folds} folds.")
        except Exception as e:
            logger.warning(f"ACC-FIX 4: Stacking meta-learner training failed: {e}. Falling back to routing mode for this fit.")
            self.meta_learner_ = None
            self._stacking_track_names_ = track_names

    def _stacking_predict_input(self, X_selected: np.ndarray) -> np.ndarray:
        """ACC-FIX 4: Build the stacked per-track feature matrix at inference time.

        Uses the PRODUCTION tracks (self.tracks, trained on the full/bootstrap data by
        _create_tracks) — only the meta-learner's *training* data came from the separate
        OOF pass in _fit_stacking; at predict time we use the real, fully-trained tracks
        for their best possible individual outputs, exactly matching how stacking ensembles
        are normally deployed.

        PHASE7: when self._stacking_signals_active_ is True (set in _fit_stacking, see its
        docstring), exactly 5 structural-signal columns are appended here too, computed from
        the SAME per-track outputs already gathered in this method (no extra track calls). If
        signal computation itself fails for a batch, a zero-filled block of the same width is
        appended instead of skipping it — the meta-learner was fit expecting this many columns,
        so the width must always match regardless of whether the signals themselves are usable.
        """
        track_names = self._stacking_track_names_ or []
        n_samples = len(X_selected)
        signals_active = getattr(self, '_stacking_signals_active_', False)
        if self.task_type == "classification":
            n_classes = len(self.classes_)
            stacked = np.zeros((n_samples, len(track_names) * n_classes))
            track_idx_preds = np.zeros((n_samples, len(track_names)))
            proba_sum = np.zeros((n_samples, n_classes))
            n_proba_ok = 0
            for i, name in enumerate(track_names):
                if name not in self.tracks:
                    # ACC-FIX 4: a track referenced at stacking-fit time was later pruned/removed;
                    # leave its meta-feature slot as zeros rather than erroring out.
                    continue
                track = self.tracks[name]
                try:
                    proba = track.predict_proba(X_selected)
                    track_classes = getattr(track.classifier, 'classes_', self.classes_)
                    proba = self._align_proba(proba, track_classes)
                    stacked[:, i * n_classes:(i + 1) * n_classes] = proba
                    track_idx_preds[:, i] = np.argmax(proba, axis=1)
                    proba_sum += proba
                    n_proba_ok += 1
                except Exception as e:
                    logger.debug(f"Stacking input: track {name} predict_proba failed: {e}")
            if signals_active:
                stacking_signals = np.zeros((n_samples, 5))
                if self.signal_extractor_ is not None and self.signal_extractor_.fitted_:
                    try:
                        ensemble_proba = proba_sum / n_proba_ok if n_proba_ok > 0 else None
                        stacking_signals = self.signal_extractor_.extract_signals(
                            X_selected, track_predictions=track_idx_preds, ensemble_proba=ensemble_proba)
                    except Exception as e:
                        logger.debug(f"PHASE7: stacking-input signal computation failed, using zeros: {e}")
                stacked = np.hstack([stacked, stacking_signals])
        else:
            stacked = np.zeros((n_samples, len(track_names)))
            for i, name in enumerate(track_names):
                if name not in self.tracks:
                    continue
                track = self.tracks[name]
                try:
                    stacked[:, i] = track.predict(X_selected)
                except Exception as e:
                    logger.debug(f"Stacking input: track {name} predict failed: {e}")
            if signals_active:
                stacking_signals = np.zeros((n_samples, 5))
                if self.signal_extractor_ is not None and self.signal_extractor_.fitted_:
                    try:
                        stacking_signals = self.signal_extractor_.extract_signals(
                            X_selected, track_predictions=stacked)
                    except Exception as e:
                        logger.debug(f"PHASE7: stacking-input signal computation failed, using zeros: {e}")
                stacked = np.hstack([stacked, stacking_signals])
        return stacked

    def _compute_expert_diagnostics(self, X_train: np.ndarray, y_train: np.ndarray,
                                     oof_out: np.ndarray, track_names: List[str]) -> Dict[str, Any]:
        """PHASE3 (spec section 4 + item 17): leak-free expert diversity + marginal-contribution
        diagnostics, computed once per fit() from the shared OOF track outputs.

        - pairwise_pred_correlation: correlation of OOF PREDICTIONS between every pair of
          tracks (classification: each slot's OOF-predicted class index; regression: raw OOF
          point predictions). High correlation = redundant experts.
        - pairwise_error_correlation: correlation of OOF LOSS between every pair of tracks
          (spec item 4A — "OOF prediction-error correlation penalty"). This is a genuinely
          distinct signal from prediction correlation: two experts can predict similar
          values/classes yet fail on different samples (low error correlation = genuinely
          complementary even if their raw predictions look alike).
        - diversity_score: 1 - mean(off-diagonal error correlation); higher means the experts
          tend to fail on different samples.
        - marginal_gain: cheap leave-one-out OOF ablation (spec item 17's "cheap approximations
          where full ablation is too expensive") — for each track, the increase in
          full-ensemble (simple-average fusion) OOF loss when that track is excluded from the
          average. No refitting needed since every track's OOF output is already available.
          Consumed by _prune_unused_tracks via Track.performance_score.
        """
        n_slots = oof_out.shape[1]
        n = len(y_train)

        if self.task_type == "classification":
            true_class_indices = np.array([
                np.where(self.classes_ == y_train[j])[0][0] if y_train[j] in self.classes_ else 0
                for j in range(n)
            ])
            per_slot_pred = np.argmax(oof_out, axis=2).astype(float)
            per_slot_loss = 1.0 - oof_out[np.arange(n)[:, None], np.arange(n_slots)[None, :], true_class_indices[:, None]]
            full_ensemble_proba = oof_out.mean(axis=1)
            full_ensemble_loss = float(np.mean(1.0 - full_ensemble_proba[np.arange(n), true_class_indices]))
        else:
            per_slot_pred = oof_out
            per_slot_loss = np.abs(oof_out - y_train[:, None])
            full_ensemble_pred = oof_out.mean(axis=1)
            full_ensemble_loss = float(np.mean(np.abs(y_train - full_ensemble_pred)))

        if n_slots > 1:
            with np.errstate(invalid='ignore'):
                pred_corr = np.nan_to_num(np.corrcoef(per_slot_pred, rowvar=False), nan=0.0)
                error_corr = np.nan_to_num(np.corrcoef(per_slot_loss, rowvar=False), nan=0.0)
            off_diag = ~np.eye(n_slots, dtype=bool)
            diversity_score = float(1.0 - error_corr[off_diag].mean())
        else:
            pred_corr = np.ones((1, 1))
            error_corr = np.ones((1, 1))
            diversity_score = 1.0

        marginal_gain = {}
        for e in range(n_slots):
            other_slots = [s for s in range(n_slots) if s != e]
            if not other_slots:
                marginal_gain[track_names[e]] = 0.0
                continue
            if self.task_type == "classification":
                loo_proba = oof_out[:, other_slots, :].mean(axis=1)
                loo_loss = float(np.mean(1.0 - loo_proba[np.arange(n), true_class_indices]))
            else:
                loo_pred = oof_out[:, other_slots].mean(axis=1)
                loo_loss = float(np.mean(np.abs(y_train - loo_pred)))
            marginal_gain[track_names[e]] = loo_loss - full_ensemble_loss  # positive = removing hurts

        names = track_names[:n_slots]
        pred_corr_by_name = {names[i]: {names[j]: float(pred_corr[i, j]) for j in range(n_slots)} for i in range(n_slots)}
        error_corr_by_name = {names[i]: {names[j]: float(error_corr[i, j]) for j in range(n_slots)} for i in range(n_slots)}

        return {
            'track_names': names,
            'pairwise_pred_correlation': pred_corr,
            'pairwise_error_correlation': error_corr,
            'pairwise_pred_correlation_by_name': pred_corr_by_name,
            'pairwise_error_correlation_by_name': error_corr_by_name,
            'mean_pairwise_error_correlation': float(error_corr[~np.eye(n_slots, dtype=bool)].mean()) if n_slots > 1 else 0.0,
            'diversity_score': diversity_score,
            'marginal_gain': marginal_gain,
            'full_ensemble_oof_loss': full_ensemble_loss,
        }

    def get_diversity_report(self) -> Dict[str, Any]:
        """PHASE3 (spec section 4): public accessor for the diversity/marginal-contribution
        diagnostics computed at fit() time. Raises if unavailable (enable_correction_track,
        router_use_oof, router_loss_aware and enable_difficulty_model were all disabled, so no
        OOF pass — and therefore no diagnostics — was computed for this fit)."""
        check_is_fitted(self)
        if self.diversity_diagnostics_ is None:
            raise ValueError("Diversity diagnostics not available for this fit (no OOF pass was "
                              "computed — enable one of enable_correction_track/router_use_oof/"
                              "router_loss_aware/enable_difficulty_model to get diagnostics).")
        return self.diversity_diagnostics_

    def _fit_loss_aware_router(self, X_train: np.ndarray, y_train: np.ndarray, oof_out: np.ndarray):
        """PHASE2 (spec section 3): train a per-track EXPECTED LOSS regressor from leak-free
        OOF track outputs, upgrading the router's training signal from a "best track"
        classification target (argmax OOF prob / argmin OOF error — see ACC-FIX 5) into a
        genuine per-expert expected-loss estimate: E[L_e | x] for every track e.

        Loss definitions (spec section 3):
            classification: 1 - p_true   (per OOF-aligned track probability of the true class)
            regression:     |y - pred|   (absolute error)

        A single multi-output RandomForestRegressor predicts all n_slots losses at once — this
        keeps the extra inference cost to one model call (not one model per track) and avoids
        the "giant neural network to look advanced" anti-pattern spec item 22 warns against.
        Consumed at inference by _apply_loss_aware_blend via self.expert_loss_model_.
        """
        try:
            track_names = [t for t in self.tracks.keys() if t != 'correction_track']
            n_slots = oof_out.shape[1]

            if self.task_type == "classification":
                true_class_indices = np.array([
                    np.where(self.classes_ == y_train[j])[0][0] if y_train[j] in self.classes_ else 0
                    for j in range(len(y_train))
                ])
                # oof_out: (n_samples, n_slots, n_classes) -> pick p(true_class) per slot.
                loss_targets = 1.0 - oof_out[np.arange(len(y_train))[:, None],
                                              np.arange(n_slots)[None, :],
                                              true_class_indices[:, None]]
            else:
                loss_targets = np.abs(oof_out - y_train[:, None])

            model = RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=3,
                random_state=self.random_state, n_jobs=self.n_jobs
            )
            model.fit(X_train, loss_targets)
            self.expert_loss_model_ = model
            self._loss_aware_track_names_ = track_names[:n_slots]
            logger.info(f"PHASE2: loss-aware expert-utility router trained "
                        f"({n_slots} tracks, mean OOF loss={float(loss_targets.mean()):.4f}).")
        except Exception as e:
            logger.warning(f"PHASE2: loss-aware router training failed, routing will fall back "
                            f"to router-prior-only weights: {e}")
            self.expert_loss_model_ = None
            self._loss_aware_track_names_ = []

    def _fit_difficulty_model(self, X_train: np.ndarray, y_train: np.ndarray, oof_out: np.ndarray):
        """PHASE2 (spec section 5): fit a leak-free sample-difficulty estimator.

        "Difficulty" here is operationalized as the ENSEMBLE's mean OOF loss on a sample —
        i.e. genuinely how hard the sample was for the (leak-free-evaluated) heterogeneous
        expert pool, not a proxy invented after the fact. We regress this OOF-derived target
        against the cheap structural signals from self.signal_extractor_ (disagreement,
        entropy, density, cluster distance, outlier score) so the estimator is usable at
        INFERENCE time, when no labels/OOF evaluation are available. Training-set difficulty
        scores are stored to convert future raw regressor output into an empirical-CDF-based
        [0, 1] score via predict_difficulty().
        """
        try:
            if self.task_type == "classification":
                true_class_indices = np.array([
                    np.where(self.classes_ == y_train[j])[0][0] if y_train[j] in self.classes_ else 0
                    for j in range(len(y_train))
                ])
                per_slot_loss = 1.0 - oof_out[np.arange(len(y_train))[:, None],
                                               np.arange(oof_out.shape[1])[None, :],
                                               true_class_indices[:, None]]
            else:
                per_slot_loss = np.abs(oof_out - y_train[:, None])
            difficulty_target = per_slot_loss.mean(axis=1)  # ensemble-mean OOF loss per sample

            track_preds_for_signals = None
            ensemble_proba = None
            if self.task_type == "regression":
                track_preds_for_signals = oof_out  # (n, n_slots) — matches SignalExtractor's expectation
            else:
                # PHASE2: classification disagreement signal — use each slot's OOF-predicted
                # class INDEX (not the probability vector) so SignalExtractor's std-based
                # disagreement signal has a numeric matrix to work with, consistent with how
                # _build_router_meta_features already treats hard class predictions as numeric
                # for its own disagreement column.
                track_preds_for_signals = np.argmax(oof_out, axis=2).astype(float)
                # PHASE7 FIX: oof_out is already class-aligned (see _compute_oof_track_outputs),
                # so its mean over the track axis is directly the ensemble probability the
                # entropy signal needs -- no separate alignment step required here.
                ensemble_proba = oof_out.mean(axis=1)
            signals = self.signal_extractor_.extract_signals(
                X_train, track_predictions=track_preds_for_signals, ensemble_proba=ensemble_proba)

            model = RandomForestRegressor(
                n_estimators=60, max_depth=6, min_samples_leaf=5,
                random_state=self.random_state, n_jobs=self.n_jobs
            )
            model.fit(signals, difficulty_target)
            self.difficulty_model_ = model
            train_scores = model.predict(signals)
            self._difficulty_train_scores_ = np.sort(train_scores)
            logger.info(f"PHASE2: difficulty model fitted (mean OOF loss={float(difficulty_target.mean()):.4f}).")
        except Exception as e:
            logger.warning(f"PHASE2: difficulty model training failed: {e}")
            self.difficulty_model_ = None
            self._difficulty_train_scores_ = None

    def _difficulty_from_selected(self, X_selected: np.ndarray) -> np.ndarray:
        """Core of predict_difficulty(), operating directly on ALREADY preprocessed/feature-
        selected data. Factored out (spec item 32) so predict()/predict_proba() can compute
        difficulty inline — for PHASE5's expected-risk abstention — without double-applying
        preprocessing or recursively calling predict() through the public predict_difficulty()
        API. See predict_difficulty() for the full explanation of what this score means.
        """
        track_preds_for_signals = None
        ensemble_proba = None
        track_names = [t for t in self.tracks.keys() if t != 'correction_track']
        if track_names:
            if self.task_type == "regression":
                preds = np.zeros((len(X_selected), len(track_names)))
                for i, name in enumerate(track_names):
                    try:
                        preds[:, i] = self.tracks[name].predict(X_selected)
                    except Exception:
                        pass
                track_preds_for_signals = preds
            else:
                idx_preds = np.zeros((len(X_selected), len(track_names)))
                # PHASE7 FIX: use predict_proba (not predict) so the SAME per-track call also
                # yields a genuine mean-ensemble probability for the entropy signal, instead of
                # entropy silently defaulting to 0 here as it always did before.
                aligned_probas = []
                for i, name in enumerate(track_names):
                    try:
                        track = self.tracks[name]
                        proba = track.predict_proba(X_selected)
                        track_classes = getattr(track.classifier, 'classes_', self.classes_)
                        proba_aligned = self._align_proba(proba, track_classes)
                        idx_preds[:, i] = np.argmax(proba_aligned, axis=1)
                        aligned_probas.append(proba_aligned)
                    except Exception:
                        pass
                track_preds_for_signals = idx_preds
                ensemble_proba = np.mean(aligned_probas, axis=0) if aligned_probas else None

        signals = self.signal_extractor_.extract_signals(
            X_selected, track_predictions=track_preds_for_signals,
            ensemble_proba=ensemble_proba if self.task_type != "regression" else None)
        raw_scores = self.difficulty_model_.predict(signals)

        if self._difficulty_train_scores_ is not None and len(self._difficulty_train_scores_) > 0:
            percentiles = np.searchsorted(self._difficulty_train_scores_, raw_scores) / len(self._difficulty_train_scores_)
            return np.clip(percentiles, 0.0, 1.0)
        return np.clip(raw_scores, 0.0, 1.0)

    def predict_difficulty(self, X) -> np.ndarray:
        """PHASE2 (spec section 5): estimate per-sample difficulty in [0, 1].

        0 = easy (structurally similar to samples the OOF-evaluated ensemble handled well),
        1 = hard (similar to samples with high OOF ensemble loss). Computed from the same
        structural signals used for routing (expert disagreement, predictive entropy, local
        density, cluster distance, outlier score) fed through a regressor trained against
        genuine leak-free OOF ensemble loss (see _fit_difficulty_model) — this is an ensemble-
        loss PROXY, not a calibrated probability of misclassification.

        Raises if enable_difficulty_model=False or too little data was available to fit one;
        callers should treat difficulty as an optional diagnostic, not a required signal.
        """
        check_is_fitted(self)
        if self.difficulty_model_ is None:
            raise ValueError("Difficulty model not available (enable_difficulty_model=False, "
                              "or there wasn't enough data to fit one during training).")
        if not isinstance(X, pd.DataFrame):
            X_arr = check_array(X, accept_sparse=False)  # allow-nan is set automatically by the check_array wrapper above
        else:
            X_arr = X
        X_scaled = self.preprocessor_.transform(X_arr)
        X_selected = self.feature_selector_.transform(X_scaled) if self.feature_selector_ is not None else X_scaled
        return self._difficulty_from_selected(X_selected)

    def _train_router(self, X_holdout: np.ndarray, y_holdout: np.ndarray,
                       best_track_indices_override: Optional[np.ndarray] = None,
                       oof_out: Optional[np.ndarray] = None):
        """Train the stronger router model (IMPROVEMENT 1).

        ACC-FIX 5: accepts an optional `best_track_indices_override` — when provided (by the
        router_use_oof path in fit()), these OOF-derived labels are used directly as the
        router's training target instead of scoring tracks on X_holdout here.

        PHASE3-FIX (spec item I): also accepts the raw `oof_out` array itself. When both are
        provided (the router_use_oof path — where X_holdout IS X_train_tracks, i.e. NOT a
        genuine holdout), the router's INPUT meta-features are now built from these same
        leak-free OOF per-track outputs instead of calling track.predict_proba(X_holdout)/
        track.predict(X_holdout) directly. The original code got the router's TRAINING LABEL
        leak-free via best_track_indices_override but still built its INPUT FEATURES from
        in-sample track predictions on X_holdout==X_train_tracks — the router's features were
        systematically over-confident/optimistic relative to what those tracks would produce on
        genuinely unseen data, which is a real (if easy-to-miss) leakage channel distinct from
        target leakage. When oof_out is None (the genuine-holdout path, router_use_oof=False),
        behavior is unchanged: X_holdout is real held-out data, so scoring tracks on it directly
        is fine.
        """
        if getattr(self, 'intelligent_routing', False):
            return self._train_hierarchical_router(X_holdout, y_holdout, best_track_indices_override, oof_out=oof_out)

        logger.info(f"Training {self.router_type} router...")
        
        n_holdout = len(X_holdout)
        track_names = list(self.tracks.keys())
        n_tracks = len(track_names)
        
        # Find best track for each sample
        best_track_indices = np.zeros(n_holdout, dtype=int)
        
        # FIX 4: cache each track's output on X_holdout here, then hand it to
        # _build_router_meta_features below so it doesn't call predict/predict_proba again
        track_output_cache = {}
        
        if best_track_indices_override is not None:
            # ACC-FIX 5: use OOF-derived best-track labels.
            best_track_indices = best_track_indices_override
            if oof_out is not None:
                # PHASE3-FIX (item I): seed the cache from the leak-free OOF outputs (already
                # aligned to self.classes_ for classification — see _compute_oof_track_outputs)
                # so _build_router_meta_features below builds its meta-features from these
                # instead of recomputing in-sample track.predict_proba(X_holdout) results.
                base_track_names = [t for t in self.tracks.keys() if t != 'correction_track']
                for i, tname in enumerate(base_track_names[:oof_out.shape[1]]):
                    track_output_cache[tname] = oof_out[:, i, :] if self.task_type == "classification" else oof_out[:, i]
            # else: track_output_cache stays empty so _build_router_meta_features below
            # computes fresh production-track outputs on X_holdout (the genuine-holdout case).
        elif self.task_type == "classification":
            track_probas = np.zeros((n_holdout, n_tracks))
            true_class_indices = np.array([
                np.where(self.classes_ == y_holdout[j])[0][0]
                if y_holdout[j] in self.classes_ else 0
                for j in range(n_holdout)
            ])
            for i, track_name in enumerate(track_names):
                track = self.tracks[track_name]
                try:
                    proba_raw = track.predict_proba(X_holdout)
                    track_output_cache[track_name] = proba_raw  # FIX 4: cache for reuse below
                    # PHASE4-FIX: a track (especially a dynamically-spawned specialist trained
                    # on a small, possibly single-class buffer subset) can return fewer columns
                    # than len(self.classes_) if it never saw every class during its own fit.
                    # Indexing proba_raw directly with a global class index then breaks with an
                    # "index out of bounds" error. _align_proba safely scatters whatever classes
                    # the track DOES know about into the full global class space first.
                    track_classes = getattr(track.classifier, 'classes_', self.classes_)
                    proba_aligned = self._align_proba(proba_raw, track_classes)
                    track_probas[:, i] = proba_aligned[np.arange(n_holdout), true_class_indices]
                except Exception as e:
                    logger.warning(f"Failed to evaluate track {track_name}: {e}")
                    track_probas[:, i] = -1.0
            best_track_indices = np.argmax(track_probas, axis=1)
        else:
            track_errors = np.zeros((n_holdout, n_tracks))
            for i, track_name in enumerate(track_names):
                track = self.tracks[track_name]
                try:
                    preds = track.predict(X_holdout)
                    track_output_cache[track_name] = preds.astype(float)  # FIX 4: cache for reuse below
                    track_errors[:, i] = np.abs(preds - y_holdout)
                except Exception as e:
                    logger.warning(f"Failed to evaluate track {track_name}: {e}")
                    track_errors[:, i] = np.inf
            best_track_indices = np.argmin(track_errors, axis=1)
        
        # Build router training data (FIX 4: reuses track_output_cache instead of recomputing)
        X_router_train = self._build_router_meta_features(X_holdout, track_output_cache=track_output_cache)
        self._router_uses_meta_features = (X_router_train.shape[1] > X_holdout.shape[1])
        
        if self._router_uses_meta_features:
            logger.info(f"Router trained with meta-features: {X_router_train.shape[1]} features")
        
        # Create and train router (PHASE7: adapt max_depth to the actual router training size)
        self.router_ = self._create_stronger_router(n_samples=n_holdout)
        
        try:
            if len(np.unique(best_track_indices)) > 1:
                self.router_.fit(X_router_train, best_track_indices)
            else:
                logger.warning("Only one track preferred. Using constant predictor.")
                from sklearn.dummy import DummyClassifier
                self.router_ = DummyClassifier(strategy='constant', constant=best_track_indices[0])
                self.router_.fit(X_router_train, best_track_indices)
        except Exception as e:
            logger.warning(f"Router training failed: {e}. Using fallback.")
            from sklearn.dummy import DummyClassifier
            self.router_ = DummyClassifier(strategy='most_frequent')
            self.router_.fit(X_router_train, best_track_indices)
        
        self.router_track_names_ = track_names
        
        unique, counts = np.unique(best_track_indices, return_counts=True)
        track_dist = {track_names[idx]: int(count) for idx, count in zip(unique, counts)}
        logger.info(f"Optimal router distribution: {track_dist}")
        
    def _prune_unused_tracks(self):
        """PHASE3 (spec item 17): contribution-aware pruning.

        Replaces the earlier usage/time-only heuristic (Track.is_underused() alone, with a
        hardcoded "never go below 2 tracks, never touch track_0") with a combined score that
        also accounts for:
          - contribution: the track's OOF leave-one-out marginal-gain estimate computed once at
            fit() time (see _compute_expert_diagnostics) and stored in Track.performance_score.
            A track that is rarely selected online but was shown, via leak-free OOF ablation, to
            meaningfully reduce ensemble loss (e.g. it specializes in an under-represented
            region) has a HIGH contribution score and is protected from pruning even while idle.
          - redundancy: mean OOF prediction correlation with the other currently-kept tracks
            (from self.diversity_diagnostics_) — a track that is both idle online AND nearly
            redundant with a surviving track is the strongest prune candidate.
        Falls back to the pure usage/time heuristic (is_underused() only) when no OOF
        diagnostics were computed for this fit (same graceful-degradation pattern used
        elsewhere when the richer signal isn't available).

        Never prunes below self.minimum_experts (configurable; default 2 matches prior
        hardcoded behavior), and never touches 'track_0' (kept as an always-available anchor)
        or the correction track (a separate pool, never part of self.tracks).
        """
        if not self.enable_track_pruning or len(self.tracks) <= self.minimum_experts:
            return

        # PHASE4 update: removing a track changes the expected input dimensionality of the
        # router (when it consumes meta-features — one column block per currently-present
        # track, see _build_router_meta_features) or the stacking meta-learner (same issue).
        # PHASE3 originally made pruning a no-op in both cases since no refresh mechanism
        # existed yet. PHASE4 adds refresh_router()/refresh_fusion() (spec item 12): for the
        # flat-router-with-meta-features case, refreshing is cheap (one router refit on
        # already-available data) so pruning now proceeds and auto-refreshes afterward. For
        # STACKING, refresh_fusion() reruns a full k-fold OOF pass — expensive, and pruning can
        # fire from inside an ordinary predict() call (see the periodic check below) — so
        # auto-refreshing there would silently make some predict() calls dramatically slower
        # (spec item 31: don't make things 10x slower without justification). Stacking mode
        # keeps the PHASE3 skip-pruning behavior; call refresh_fusion() manually when it's a
        # good time to pay that cost.
        if getattr(self, '_active_combination_mode_', None) == "stacking":
            logger.debug("Track pruning skipped this cycle: stacking mode's meta-learner "
                          "dimensionality depends on the current track count, and refreshing it "
                          "requires a full OOF pass — too expensive to trigger automatically "
                          "from inside predict(). Call refresh_fusion() explicitly when ready.")
            return
        needs_router_refresh_after = getattr(self, '_router_uses_meta_features', False)

        diagnostics = self.diversity_diagnostics_
        candidates = []
        for track_name, track in self.tracks.items():
            if track_name == 'track_0':
                continue
            if not track.is_underused():
                continue  # only ever prune tracks the online router has stopped relying on

            if diagnostics is not None and track_name in diagnostics.get('pairwise_pred_correlation_by_name', {}):
                contribution = track.performance_score
                corr_row = diagnostics['pairwise_pred_correlation_by_name'][track_name]
                other_corrs = [v for k, v in corr_row.items() if k != track_name and k in self.tracks]
                redundancy = float(np.mean(other_corrs)) if other_corrs else 0.0
            else:
                contribution = 0.0
                redundancy = 0.0
            # Lower contribution + higher redundancy => stronger prune candidate.
            prune_score = redundancy - contribution
            candidates.append((prune_score, track_name, contribution, redundancy))

        if not candidates:
            return

        candidates.sort(key=lambda c: c[0], reverse=True)  # most prunable first
        max_removable = len(self.tracks) - self.minimum_experts
        tracks_to_remove = [name for _, name, _, _ in candidates[:max_removable]]

        if not tracks_to_remove:
            return

        for track_name in tracks_to_remove:
            contribution = self.tracks[track_name].performance_score
            del self.tracks[track_name]
            logger.info(f"Pruned track: {track_name} (contribution-aware; OOF marginal "
                        f"contribution={contribution:.4f})")

        if needs_router_refresh_after:
            # PHASE4 (spec item 12): the router's fitted meta-feature dimensionality no longer
            # matches the (now smaller) track set — retrain it against the current tracks using
            # the same data it was last trained on, rather than leaving a broken model behind.
            logger.info("Refreshing router after pruning to match the new track set...")
            self.refresh_router()

    def refresh_router(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """PHASE4 (spec item 12): retrain the router (flat or hierarchical) against the
        CURRENT contents of self.tracks. Needed whenever the track pool's composition changes
        after the router was last trained — most notably after dynamic spawning accepts a new
        specialist (called automatically there; see _check_spawn_new_track) — since the flat
        router's meta-features and the hierarchical stack's regional membership both depend on
        exactly which tracks currently exist.

        If X/y aren't supplied, reuses the data captured at the end of the last fit()/
        partial_fit() call (self._last_router_X_/_last_router_y_) — i.e. calling
        refresh_router() with no arguments repeats the original router-training procedure with
        whatever tracks exist right now. No-op if there's no router-training data on record or
        fewer than 2 tracks (nothing to route between).
        """
        X_ref = X if X is not None else self._last_router_X_
        y_ref = y if y is not None else self._last_router_y_
        if X_ref is None or y_ref is None or len(self.tracks) < 2:
            logger.debug("refresh_router: no router-training data on record, or fewer than 2 "
                          "tracks — nothing to refresh.")
            return
        self._train_router(X_ref, y_ref)
        self._active_combination_mode_ = "routing"
        self._last_router_X_, self._last_router_y_ = X_ref, y_ref
        logger.info(f"refresh_router: router retrained against {len(self.tracks)} current tracks.")

    def refresh_fusion(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """PHASE4 (spec item 12): retrain the stacking meta-learner against the CURRENT
        contents of self.tracks, using a fresh (leak-free) OOF pass on the supplied or
        most-recently-seen router-training data. Deliberately NOT a full retrain over the
        model's entire training history — spec item 13 explicitly warns against retraining
        everything on every update — just enough data to safely re-fit the meta-learner's
        input dimensionality and weights for the current track set.
        """
        X_ref = X if X is not None else self._last_router_X_
        y_ref = y if y is not None else self._last_router_y_
        if X_ref is None or y_ref is None or len(self.tracks) < 2:
            logger.debug("refresh_fusion: no data on record, or fewer than 2 tracks — nothing to refresh.")
            return
        self._fit_stacking(X_ref, y_ref)
        if self.meta_learner_ is not None:
            self.router_ = None
            self.router_track_names_ = list(self.tracks.keys())
            self._active_combination_mode_ = "stacking"
            self._last_router_X_, self._last_router_y_ = X_ref, y_ref
            logger.info(f"refresh_fusion: stacking meta-learner retrained against {len(self.tracks)} current tracks.")
        else:
            logger.warning("refresh_fusion: stacking refit failed; falling back to refresh_router() "
                            "so predict()/predict_proba() still have a usable combination path.")
            self.refresh_router(X_ref, y_ref)

    def refresh_experts(self):
        """PHASE4 (spec item 12): recompute per-expert diagnostics (contribution/redundancy —
        see _compute_expert_diagnostics) against the CURRENT track set and current
        self._last_router_X_/_last_router_y_. Useful after dynamic spawning or any manual
        change to self.tracks, so pruning decisions reflect the present track pool rather than
        a stale snapshot from the original fit(). No-op if there's no router-training data on
        record or fewer than 2 tracks.
        """
        if self._last_router_X_ is None or self._last_router_y_ is None or len(self.tracks) < 2:
            logger.debug("refresh_experts: no data on record, or fewer than 2 tracks — nothing to refresh.")
            return
        try:
            oof_out = self._compute_oof_track_outputs(self._last_router_X_, self._last_router_y_, self.stacking_folds)
            track_names_for_diag = [t for t in self.tracks.keys() if t != 'correction_track']
            self.diversity_diagnostics_ = self._compute_expert_diagnostics(
                self._last_router_X_, self._last_router_y_, oof_out, track_names_for_diag)
            for name, gain in self.diversity_diagnostics_['marginal_gain'].items():
                if name in self.tracks:
                    self.tracks[name].performance_score = gain
            logger.info(f"refresh_experts: diagnostics recomputed for {len(track_names_for_diag)} current tracks.")
        except Exception as e:
            logger.warning(f"refresh_experts failed: {e}")
    
    def fit(self, X, y):
        """Fit the enhanced TRA model."""
        # IMPROVEMENT: A refit starts with fresh learned state, including after set_params().
        fresh = type(self)(**self.get_params(deep=False))
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        start_time = time.time()
        logger.info("Fitting Enhanced Optimized TRA model...")
        
        # PHASE4: a fresh fit() call means a fresh model — reset all online-adaptation state
        # (spawn buffer/cooldown, drift detector) rather than carrying it over from any
        # previous fit()/partial_fit() history on this same estimator instance.
        self._uncertain_buffer_X_ = None
        self._uncertain_buffer_y_ = None
        self._last_spawn_step_ = -10**9
        self._call_step_ = 0
        self._drift_detector_ = PageHinkleyDetector(threshold=self.drift_threshold)
        self.drift_detected_ = False
        self._drift_event_count_ = 0
        self._abstention_count_ = 0
        self._correction_applied_count_ = 0
        self._routing_history_ = deque(maxlen=2000)
        self._inference_latencies_ = deque(maxlen=500)
        
        # Validate input
        if not isinstance(X, pd.DataFrame):
            X, y = check_X_y(X, y, accept_sparse=False)  # allow-nan is set automatically by the check_X_y wrapper above
        else:
            y = np.asarray(y)
            
        self.n_features_in_ = X.shape[1]
        # PHASE6 item 29: capture feature names when available (standard sklearn convention
        # attribute name) so save_model()/load_model() can record and validate against them.
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        
        # PHASE1 item 7: carve off a genuinely held-out conformal calibration slice BEFORE any
        # preprocessing/track/router/stacking training happens, so it is never seen by anything
        # this model fits — required for predict_interval()'s coverage semantics to mean
        # anything. Only for regression (classification calibration is handled separately via
        # CalibratedClassifierCV elsewhere) and only when there's enough data to spare.
        X_calib_raw, y_calib_raw = None, None
        if (self.task_type == "regression" and self.enable_conformal
                and len(X) >= 2 * self.conformal_min_samples):
            try:
                X, X_calib_raw, y, y_calib_raw = train_test_split(
                    X, y, test_size=self.conformal_calibration_fraction, random_state=self.random_state
                )
                logger.info(f"PHASE1: carved off {len(X_calib_raw)} samples for conformal "
                            f"calibration (never used for track/router/stacking training).")
            except ValueError as e:
                logger.debug(f"Conformal calibration split skipped: {e}")
                X_calib_raw, y_calib_raw = None, None
        
        # PHASE5 item 21: carve off a separate genuinely held-out slice for ensemble-level
        # OUTPUT calibration (classification only) — same "never seen by anything this model
        # fits" principle as the conformal slice above, kept as an independent split so the two
        # calibration mechanisms (regression intervals vs classification probability
        # calibration) never share or contend for the same held-out rows.
        X_out_calib_raw, y_out_calib_raw = None, None
        if (self.task_type == "classification" and self.calibrate_output
                and len(X) >= 2 * self.calibration_min_samples):
            try:
                X, X_out_calib_raw, y, y_out_calib_raw = train_test_split(
                    X, y, test_size=self.calibration_fraction, random_state=self.random_state,
                    stratify=y if len(np.unique(y)) > 1 else None
                )
                logger.info(f"PHASE5: carved off {len(X_out_calib_raw)} samples for output "
                            f"calibration (never used for track/router/stacking training).")
            except ValueError as e:
                logger.debug(f"Output calibration split skipped: {e}")
                X_out_calib_raw, y_out_calib_raw = None, None
        
        # Store classes for classification
        if self.task_type == "classification":
            self.classes_ = np.unique(y)
            logger.info(f"Found {len(self.classes_)} classes: {self.classes_}")
        
        # Handle class imbalance
        self._handle_class_imbalance(y)
        
        # Setup and apply preprocessing
        self._setup_preprocessing(X)
        X_scaled = self.preprocessor_.fit_transform(X)
        
        # Enhanced feature selection
        self._setup_feature_selection(X_scaled, y)
        if self.feature_selector_ is not None:
            X_selected = self.feature_selector_.transform(X_scaled)
        else:
            X_selected = X_scaled
            
        # ACC-FIX 6: small_data_auto_mode — auto-apply data-efficient defaults on small datasets.
        # Uses local "effective_*" variables (rather than mutating self.bagging_mode /
        # self.combination_mode) so the constructor hyperparameters stay untouched and
        # get_params()/clone() keep reflecting what the user actually configured; only a track
        # of what was ACTUALLY used gets persisted (self._active_combination_mode_) for
        # predict()/predict_proba() to consult. Per-parameter: only params still at their class
        # default get auto-switched — anything the user explicitly set away from default is
        # respected as-is (e.g. an explicit cluster_experts=True is never silently overridden).
        effective_bagging_mode = self.bagging_mode
        effective_combination_mode = self.combination_mode
        if self.combination_mode == "auto":
            # ACC2-FIX 1: validation-based mode selector replaces the small_data_auto_mode
            # heuristic when the user opts into combination_mode="auto". The legacy
            # small_data_auto_mode/small_data_threshold path below is intentionally skipped in
            # this branch — it remains active ONLY when combination_mode is explicitly "routing"
            # or "stacking" (see the elif below), per its original semantics.
            effective_combination_mode = self._select_combination_mode_auto(X_selected, y)
            logger.info(f"ACC2-FIX 1: combination_mode='auto' resolved to "
                        f"'{effective_combination_mode}' for this fit() call.")
        elif self.small_data_auto_mode and len(X_selected) < self.small_data_threshold:
            # ACC-FIX 6: small_data_auto_mode — auto-apply data-efficient defaults on small datasets.
            # Uses local "effective_*" variables (rather than mutating self.bagging_mode /
            # self.combination_mode) so the constructor hyperparameters stay untouched and
            # get_params()/clone() keep reflecting what the user actually configured; only a track
            # of what was ACTUALLY used gets persisted (self._active_combination_mode_) for
            # predict()/predict_proba() to consult. Per-parameter: only params still at their class
            # default get auto-switched — anything the user explicitly set away from default is
            # respected as-is (e.g. an explicit cluster_experts=True is never silently overridden).
            auto_applied = []
            if self.bagging_mode == "bootstrap":
                effective_bagging_mode = "full"
                auto_applied.append("bagging_mode='full'")
            if self.combination_mode == "routing":
                effective_combination_mode = "stacking"
                auto_applied.append("combination_mode='stacking'")
            if auto_applied:
                logger.info(f"ACC-FIX 6: small_data_auto_mode engaged ({len(X_selected)} samples < "
                            f"{self.small_data_threshold} threshold) — auto-applying {', '.join(auto_applied)} "
                            f"for this fit() call (cluster_experts left as configured: {self.cluster_experts}).")
        
        # FIX: DATA LEAKAGE PREVENTION / ACC-FIX 3 / ACC-FIX 4 / ACC-FIX 5
        # Split data before training tracks so router learns on completely unseen data —
        # UNLESS this fit is using stacking or OOF-based router training, in which case the
        # unbiased training signal comes from k-fold OOF evaluation instead, so we skip the
        # holdout carve-out entirely and let tracks train on the FULL selected data (this is
        # the direct fix for the "router is the most data-starved component" diagnosis).
        # IMPROVEMENT: Flat fusion has no router and needs no unused router holdout.
        if effective_combination_mode in ("stacking", "flat_average") or (effective_combination_mode == "routing" and self.router_use_oof):
            # ACC-FIX 4 / ACC-FIX 5: no holdout carve-out — see comment above.
            X_train_tracks, X_holdout_router = X_selected, X_selected
            y_train_tracks, y_holdout_router = y, y
        elif len(X_selected) > 50:
            try:
                X_train_tracks, X_holdout_router, y_train_tracks, y_holdout_router = train_test_split(
                    X_selected, y, test_size=self.router_holdout_fraction, random_state=self.random_state  # ACC-FIX 3: configurable holdout fraction (was hardcoded 0.2)
                )
            except ValueError:
                X_train_tracks, X_holdout_router = X_selected, X_selected
                y_train_tracks, y_holdout_router = y, y
        else:
             X_train_tracks, X_holdout_router = X_selected, X_selected
             y_train_tracks, y_holdout_router = y, y
        
        # Create tracks (ACC-FIX 1: pass the effective bagging_mode for this fit call)
        self._create_tracks(X_train_tracks, y_train_tracks, bagging_mode_override=effective_bagging_mode)
        
        # SIGNAL-GUIDED ROUTING: Initialize and fit signal extractor
        # (fitted in both routing and stacking modes for compatibility — see _fit_stacking's
        # docstring for why the stacking path doesn't currently consume it directly)
        self.signal_extractor_ = SignalExtractor(n_neighbors=min(5, len(X_train_tracks)-1),
                                                 random_state=self.random_state)
        self.signal_extractor_.fit(X_train_tracks, self.kmeans_)
        logger.info("Signal extraction layer fitted (disagreement, entropy, density, cluster distance, outlier)")
        
        # PHASE1: compute ONE shared leak-free OOF track-output pass upfront whenever more than
        # one downstream consumer needs it (correction track / stacking / OOF router training),
        # instead of each of _add_correction_track / _fit_stacking / the router-OOF branch below
        # independently re-running the (expensive, k-fold x n_tracks) OOF procedure on the exact
        # same (X_train_tracks, y_train_tracks). This directly addresses spec item 31 (avoid
        # redundant OOF recomputation) without changing what any consumer actually does with it.
        needs_oof = len(self.tracks) > 1 and (
            self.enable_correction_track
            or effective_combination_mode == "stacking"
            or self.router_use_oof
            or self.router_loss_aware
            or self.enable_difficulty_model
        )
        shared_oof_out = None
        if needs_oof:
            logger.info(f"Computing shared {self.stacking_folds}-fold OOF track outputs "
                        f"(reused across correction track / stacking / router as applicable)...")
            shared_oof_out = self._compute_oof_track_outputs(X_train_tracks, y_train_tracks, self.stacking_folds)

        # IMPROVEMENT 3: Add Residual Correction Track (TRA-Boost)
        if self.enable_correction_track and len(self.tracks) > 1:
            self._add_correction_track(X_train_tracks, y_train_tracks, oof_out=shared_oof_out)
        
        # PHASE2 item 3: loss-aware expert-utility router (opt-in). Trained here (rather than
        # only inside the router-OOF branch below) so it's available regardless of which
        # combination_mode is ultimately resolved — harmless extra cost when stacking ends up
        # active, since predict()/predict_proba()'s stacking path never calls _route_with_router.
        if self.router_loss_aware and len(self.tracks) > 1 and shared_oof_out is not None:
            self._fit_loss_aware_router(X_train_tracks, y_train_tracks, shared_oof_out)
        else:
            self.expert_loss_model_ = None
            self._loss_aware_track_names_ = []

        # PHASE2 item 5: leak-free sample difficulty model (opt-in, on by default — cheap).
        if self.enable_difficulty_model and len(self.tracks) > 1 and shared_oof_out is not None:
            self._fit_difficulty_model(X_train_tracks, y_train_tracks, shared_oof_out)
        else:
            self.difficulty_model_ = None
            self._difficulty_train_scores_ = None

        # PHASE3 (spec section 4 + item 17): leak-free diversity/marginal-contribution
        # diagnostics, reusing the same shared OOF pass. Populates Track.performance_score for
        # every base track, which _prune_unused_tracks then uses for contribution-aware pruning.
        self.diversity_diagnostics_ = None
        if len(self.tracks) > 1 and shared_oof_out is not None:
            try:
                track_names_for_diag = [t for t in self.tracks.keys() if t != 'correction_track']
                self.diversity_diagnostics_ = self._compute_expert_diagnostics(
                    X_train_tracks, y_train_tracks, shared_oof_out, track_names_for_diag)
                for name, gain in self.diversity_diagnostics_['marginal_gain'].items():
                    if name in self.tracks:
                        self.tracks[name].performance_score = gain
                logger.info(f"PHASE3: expert diagnostics computed — diversity_score="
                            f"{self.diversity_diagnostics_['diversity_score']:.4f}, "
                            f"mean marginal gain={float(np.mean(list(self.diversity_diagnostics_['marginal_gain'].values()))):.4f}")
            except Exception as e:
                logger.warning(f"PHASE3: expert diagnostics computation failed: {e}")
                self.diversity_diagnostics_ = None
        
        # PHASE3-G2: flat_average needs no router and no meta-learner. Tracks are already
        # trained at this point, so the fusion layer is simply skipped. router_track_names_ is
        # still populated because diagnostics, pruning and refresh_* all index through it.
        if effective_combination_mode == "flat_average":
            self.router_ = None
            self.meta_learner_ = None
            self.router_track_names_ = [t for t in self.tracks.keys() if t != 'correction_track']
            self._active_combination_mode_ = "flat_average"
            logger.info(f"PHASE3-G2: flat_average fusion — uniform average over "
                        f"{len(self.router_track_names_)} expert tracks (no router, no meta-learner).")
        # ACC-FIX 4: route to stacking meta-learner training instead of router training
        elif effective_combination_mode == "stacking":
            if len(self.tracks) > 1:
                self._fit_stacking(X_train_tracks, y_train_tracks, oof_out=shared_oof_out)
            else:
                self.meta_learner_ = None
                self._stacking_track_names_ = list(self.tracks.keys())
            self.router_ = None
            self.router_track_names_ = list(self.tracks.keys())
            self._active_combination_mode_ = "stacking" if self.meta_learner_ is not None else "routing"
            if self.meta_learner_ is None:
                # Stacking failed or wasn't viable (e.g. <2 tracks) — fall back to routing so
                # predict()/predict_proba() still have a usable combination path.
                logger.warning("ACC-FIX 4: stacking unavailable for this fit; falling back to router-based routing.")
                if len(self.tracks) > 1:
                    self._train_router(X_holdout_router, y_holdout_router)
        # Train MoE Router (ACC-FIX 5: optionally on k-fold OOF labels instead of the holdout split)
        elif len(self.tracks) > 1:
            if self.router_use_oof:
                logger.info(f"ACC-FIX 5: using shared {self.stacking_folds}-fold OOF best-track labels "
                            f"across the full training set for router training...")
                oof_out = shared_oof_out if shared_oof_out is not None else self._compute_oof_track_outputs(
                    X_train_tracks, y_train_tracks, self.stacking_folds)
                if self.task_type == "classification":
                    true_class_indices = np.array([
                        np.where(self.classes_ == y_train_tracks[j])[0][0]
                        if y_train_tracks[j] in self.classes_ else 0
                        for j in range(len(y_train_tracks))
                    ])
                    # ACC-FIX 5: best slot = highest OOF probability assigned to the true class.
                    # NOTE: only the base heterogeneous track slots (0..n_tracks-1, matching
                    # _create_tracks' rotation) participate in this OOF comparison — the separately
                    # trained correction_track isn't part of the OOF slot rotation, so it's never
                    # OOF-selected here; it still applies via the existing post-hoc residual
                    # correction step in predict()/predict_proba(), unaffected by this change.
                    best_track_indices_override = np.argmax(
                        oof_out[np.arange(len(y_train_tracks)), :, true_class_indices], axis=1
                    )
                else:
                    errors = np.abs(oof_out - y_train_tracks[:, None])
                    best_track_indices_override = np.argmin(errors, axis=1)
                self._train_router(X_holdout_router, y_holdout_router, best_track_indices_override=best_track_indices_override, oof_out=oof_out)
            else:
                self._train_router(X_holdout_router, y_holdout_router)
            self._active_combination_mode_ = "routing"
        else:
            self.router_ = None
            self.router_track_names_ = list(self.tracks.keys())
            self._active_combination_mode_ = "routing"
        
        # PHASE4 (spec item 12): remember what data trained the router/fusion layer this fit,
        # so refresh_router()/refresh_fusion() can retrain against a CHANGED track set later
        # (e.g. after dynamic spawning) without the caller needing to re-supply training data.
        self._last_router_X_ = X_train_tracks if effective_combination_mode == "stacking" else X_holdout_router
        self._last_router_y_ = y_train_tracks if effective_combination_mode == "stacking" else y_holdout_router
        
        training_time = time.time() - start_time
        logger.info(f"Enhanced training completed in {training_time:.2f}s")
        
        self.fitted_ = True
        
        # PHASE1 item 7: fit split-conformal calibration on the held-out slice carved off above.
        # Uses self.predict() (the model's real, fully-trained inference path) on data none of
        # the tracks/router/stacking/correction-track ever saw, so the resulting residual
        # distribution is a leak-free basis for predict_interval()'s coverage.
        if X_calib_raw is not None and self.task_type == "regression":
            try:
                calib_preds = np.asarray(self.predict(X_calib_raw), dtype=float)
                residuals_calib = np.abs(np.asarray(y_calib_raw, dtype=float) - calib_preds)
                # PHASE7-FIX: when abstention_threshold > 0 is also active (e.g.
                # abstention_use_difficulty=True), self.predict() can legitimately return NaN
                # for abstained calibration rows. An unfiltered NaN here silently poisons
                # np.sort()/np.quantile() in predict_interval() — NaNs sort to the end of the
                # array, landing exactly in the high-quantile region predict_interval() reads
                # from for typical confidence levels, so EVERY subsequent predict_interval()
                # call would return NaN bounds for EVERY sample, not just abstained ones.
                finite_mask = np.isfinite(residuals_calib)
                n_dropped = int((~finite_mask).sum())
                residuals_calib = residuals_calib[finite_mask]
                if len(residuals_calib) == 0:
                    raise ValueError("all calibration predictions were abstained; no finite residuals to calibrate on")
                self._conformal_residuals_ = np.sort(residuals_calib)
                self._conformal_calibrated_ = True
                extra = f" ({n_dropped} abstained calibration rows excluded)" if n_dropped else ""
                logger.info(f"Conformal calibration fitted on {len(residuals_calib)} held-out "
                            f"samples (mean abs residual={residuals_calib.mean():.4f}){extra}.")
            except Exception as e:
                logger.warning(f"Conformal calibration failed, predict_interval will fall back "
                                f"to an uncalibrated ensemble-disagreement interval: {e}")
                self._conformal_residuals_ = None
                self._conformal_calibrated_ = False
        else:
            self._conformal_residuals_ = None
            self._conformal_calibrated_ = False
        
        # PHASE5 item 21: fit ensemble-level output calibration on the held-out slice carved
        # off above. Must run AFTER self.fitted_ = True (needs a working predict_proba()) and
        # BEFORE self._calibration_method_ is set to anything — _fit_output_calibration relies
        # on that to get RAW (uncalibrated) probabilities from predict_proba() for fitting.
        self._calibration_method_ = None
        self._calibration_params_ = None
        self.calibration_report_ = None
        if X_out_calib_raw is not None and self.task_type == "classification":
            self._fit_output_calibration(X_out_calib_raw, y_out_calib_raw)
        
        return self
        
    def partial_fit(self, X, y, classes=None):
        """Out-of-core learning: train new tracks on data chunks and update the router.
        
        This makes the TRA algorithm uniquely robust for streaming data! 
        It trains K NEW tracks on the incoming chunk and then re-trains the MoE router 
        on the new data to evaluate ALL existing tracks (old and new). This gracefully 
        handles concept drift since the router learns if a new track structure fits the
        new data better than old tracks, dynamically routing samples to the most recent 
        expertise context. Track Pruning handles memory overflow over time.
        """
        start_time = time.time()
        logger.info(f"Partial fitting Enhanced TRA model on {X.shape[0] if hasattr(X, 'shape') else len(X)} samples...")
        
        if not self.fitted_:
            # First chunk
            if classes is not None and self.task_type == "classification":
                self.classes_ = np.asarray(classes)
            return self.fit(X, y)
            
        # Validate input
        if not isinstance(X, pd.DataFrame):
            X, y = check_X_y(X, y, accept_sparse=False)  # allow-nan is set automatically by the check_X_y wrapper above
        else:
            y = np.asarray(y)
            
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but TRA was fitted with {self.n_features_in_} features")
            
        # Ensure classes didn't change entirely
        # (Though we'll still support unseen classes if the base estimators are retrained to handle them,
        # but the old tracks might act wildly. Handled by Router.)
        
        # Process new chunk
        self._handle_class_imbalance(y)
        
        # Apply preprocessing (don't fit, just transform unless missing cols)
        # FIX 8: previously silently fell back to fit_transform() here, which re-fits the
        # scaler/imputer on just this chunk and silently invalidates the feature-scaling
        # assumptions every previously-trained track relies on. Raise a clear error instead
        # so streaming callers find out immediately rather than getting silently-corrupted
        # feature consistency across chunks.
        try:
            X_scaled = self.preprocessor_.transform(X)
        except Exception as e:
            logger.error(
                f"Preprocessor transform failed during partial_fit: {e}. Refusing to silently "
                "refit the preprocessor, since that would invalidate feature-scaling consistency "
                "for tracks trained on earlier chunks."
            )
            raise RuntimeError(
                f"partial_fit preprocessing failed: {e}. The new chunk's features are likely "
                "inconsistent with the schema/dtypes the preprocessor was originally fitted on "
                "in fit(). Silently refitting was intentionally removed (see FIX 8) because it "
                "breaks feature-scaling consistency across streaming chunks; ensure the incoming "
                "chunk matches the original training schema."
            ) from e
        
        if self.feature_selector_ is not None:
            X_selected = self.feature_selector_.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        self._call_step_ += 1
        
        # PHASE4 (spec items 11, 24): evaluate the PRE-UPDATE model against this new chunk
        # before any mutation — this is what "concept drift" and "where is the current
        # ensemble weak" actually mean: how the model-as-it-stands-now handles data it hasn't
        # adapted to yet. Real labels are available here (this is partial_fit's whole point),
        # so this feeds both the drift detector and the SAFE, label-driven half of dynamic
        # spawning (see _check_spawn_new_track) — no self-training risk on this path.
        try:
            old_predictions = self.predict(X)
            if self.task_type == "classification":
                old_confidences = np.max(self.predict_proba(X), axis=1)
                chunk_loss = float(np.mean(np.asarray(old_predictions) != y))
            else:
                abs_err = np.abs(y - np.asarray(old_predictions, dtype=float))
                chunk_loss = float(np.mean(abs_err))
                # Heuristic, chunk-relative "confidence" for spawn-eligibility purposes only —
                # NOT a calibrated probability. See predict_interval()/_compute_ensemble_
                # disagreement for the model's actual, validated regression uncertainty API.
                denom = float(abs_err.max()) + 1e-8
                old_confidences = 1.0 - (abs_err / denom)

            if self._drift_detector_.update(chunk_loss):
                self.drift_detected_ = True
                self._drift_event_count_ += 1
                logger.warning(f"PHASE4: concept drift detected (chunk loss={chunk_loss:.4f}); "
                                f"specialist-spawning gain threshold relaxed for this chunk, and "
                                f"router/fusion will be refreshed after this update.")
            else:
                self.drift_detected_ = False

            if self.enable_dynamic_spawning:
                self._check_spawn_new_track(X_selected, old_predictions, old_confidences,
                                             spawn_confidence_labels=y)
        except Exception as e:
            logger.debug(f"PHASE4: pre-update drift/spawn evaluation skipped: {e}")
            
        n_existing_tracks = len(self.tracks)
        logger.info(f"Adding {self.n_tracks} new tracks for the chunk...")
        n_samples = X_selected.shape[0]
        # FIX 5: use the full heterogeneous model list (cycled by track count) instead of
        # _get_base_estimator(), which always returned index 0 (e.g. always RandomForest) —
        # every dynamically streamed track was silently the same model type.
        models = self._create_heterogeneous_models(n_samples=n_samples)  # FIX 6: adapt n_estimators too
        
        for i in range(self.n_tracks):
            track_name = f"track_{n_existing_tracks + i}"
            
            # Simple bootstrap for chunk data
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_track = X_selected[indices]
            y_track = y[indices]
            
            # Feature dropout for track diversity
            n_features = X_selected.shape[1]
            if self.feature_selection and n_features > 3 and self.feature_dropout_rate != 1.0:
                # ACC-FIX 2: fixed keep-fraction when set, else preserve the exact original random range
                keep_frac = np.random.uniform(0.7, 0.9) if self.feature_dropout_rate is None else self.feature_dropout_rate
                n_select = max(2, int(n_features * keep_frac))
                feature_indices = np.random.choice(n_features, size=n_select, replace=False)
                feature_indices.sort()
                X_track_clf = X_track[:, feature_indices]
            else:
                feature_indices = None
                X_track_clf = X_track
                
            # FIX 5: cycle through model types based on total track count so far, same as _create_tracks
            clf = self._clone_track_model(models[(n_existing_tracks + i) % len(models)])
            self._fit_track_model(clf, X_track_clf, y_track)
            
            track = Track(track_name, clf, feature_indices=feature_indices)
            self.tracks[track_name] = track
            
        # PHASE4-FIX (spec item G/13): refresh whichever combination path is actually active,
        # not just self.router_. Previously partial_fit only ever retrained self.router_ — if
        # this model was fit with combination_mode="stacking", that retrained router sat unused
        # (predict()/predict_proba() consult self._active_combination_mode_, which stayed
        # "stacking" and unrefreshed) while the meta-learner silently never saw the new/streamed
        # tracks. refresh_fusion()/refresh_router() both retrain against the CURRENT track set
        # using this chunk's data, and refresh_fusion() also updates self._active_combination_mode_.
        if self._active_combination_mode_ == "stacking":
            self.refresh_fusion(X_selected, y)
        else:
            self.refresh_router(X_selected, y)
        self.refresh_experts()
        
        partial_time = time.time() - start_time
        logger.info(f"Partial fit complete in {partial_time:.2f}s. Total tracks now: {len(self.tracks)}")
        return self
    
    # ==================================================================================
    # PHASE2 (spec item 33-N): ONE routing/fusion path shared by predict() and
    # predict_proba().
    #
    # Before this refactor these were two independent ~300-line implementations of the same
    # pipeline, and they disagreed in three places that no test covered:
    #
    #   D1. CALIBRATION. predict_proba() applied _apply_output_calibration(); predict() did
    #       not. Temperature scaling happens to preserve argmax, but sigmoid/isotonic
    #       calibration is fitted per class and does NOT, so with calibrate_output=True and a
    #       validation-selected non-temperature method, predict(X) could return a different
    #       label than argmax(predict_proba(X)).
    #
    #   D2. INTELLIGENT ROUTING + SOFT MODE. In predict_proba() the intelligent-routing branch
    #       accumulated a full weighted fusion into `probabilities`, and then — because
    #       `router_probas` is None on that path, so the `routing_mode == "soft"` guard could
    #       never fire — control fell through to the hard-routing branch, which OVERWROTE every
    #       row with a single track's probabilities. The weighted fusion was dead compute, and
    #       hierarchical routing silently never performed soft fusion there. predict() had no
    #       such guard and DID honour soft mode on the same branch. So the two methods used
    #       genuinely different fusion rules for the same configuration.
    #
    #   D3. ROUTER FALLBACK BLEND. predict() reconstructed a ONE-HOT vector from the already-
    #       collapsed label (np.searchsorted over classes_) and blended that toward the
    #       all-track average, discarding the real fused distribution; predict_proba() blended
    #       the actual probabilities. Different inputs to the same nominal operation.
    #
    # The fix is structural rather than a set of patches: fusion happens exactly once, in
    # probability space for classification, and predict() is defined as argmax over the same
    # distribution predict_proba() returns. Agreement is now a property of the architecture
    # instead of something that has to be re-checked after every change.
    #
    # Bookkeeping (routing history, prediction_count_, latency, pruning, spawn checks) stays on
    # predict() alone, as before, so that calling predict_proba() does not double-count.
    # ==================================================================================

    # ==================================================================================
    # PHASE5 (spec item 30): sklearn estimator-type resolution.
    #
    # EnhancedTRA inherits BOTH ClassifierMixin and RegressorMixin so that one class can serve
    # either task. Under scikit-learn's tag system (1.6+) those two mixins each set
    # `estimator_type` in the __sklearn_tags__ chain, and the second overwrites the first — the
    # net result was `estimator_type = None`, so `is_classifier(tra)` returned False.
    #
    # The consequence was not cosmetic: every probability-based scorer refuses to run against an
    # estimator that is not a classifier, so
    #     cross_validate(tra, X, y, scoring="neg_log_loss")
    # failed with "Got a regressor with response_method=predict_proba", and silently recorded
    # NaN unless error_score="raise" was set. Log loss, Brier, ECE, GridSearchCV over any
    # probabilistic objective, and CalibratedClassifierCV wrapping were all unavailable.
    #
    # Resolving the type from self.task_type (which is a constructor parameter, so it is set
    # before any tag lookup) makes TRA report the type it actually is for this instance.
    # ==================================================================================

    @property
    def _estimator_type(self):
        """Legacy sklearn (<1.6) estimator-type hook."""
        return "classifier" if self.task_type == "classification" else "regressor"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        is_clf = self.task_type == "classification"
        try:
            tags.estimator_type = "classifier" if is_clf else "regressor"
            if is_clf:
                if getattr(tags, "classifier_tags", None) is None:
                    from sklearn.utils import ClassifierTags
                    tags.classifier_tags = ClassifierTags()
                tags.regressor_tags = None
            else:
                if getattr(tags, "regressor_tags", None) is None:
                    from sklearn.utils import RegressorTags
                    tags.regressor_tags = RegressorTags()
                tags.classifier_tags = None
        except Exception:
            # Older/newer sklearn with a different tag object: leave whatever super() produced
            # rather than raising during a routine introspection call.
            pass
        return tags

    def __sklearn_is_fitted__(self):
        # IMPROVEMENT: A constructor-created fitted_=False is not a trained estimator.
        return bool(self.fitted_)

    def _prepare_features(self, X) -> np.ndarray:
        """Validate raw input and transform it into the model's selected feature space.

        Shared by predict/predict_proba/predict_interval/explain so that input validation is
        identical everywhere. predict_proba() previously skipped the n_features_in_ check that
        predict() performed, meaning a wrong-width matrix raised a confusing downstream error
        from inside a track instead of a clear one here.
        """
        if not isinstance(X, pd.DataFrame):
            X = check_array(X, accept_sparse=False)  # allow-nan handled by the wrapper above
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but TRA was fitted with "
                             f"{self.n_features_in_} features")
        X_scaled = self.preprocessor_.transform(X)
        if self.feature_selector_ is not None:
            return self.feature_selector_.transform(X_scaled)
        return X_scaled

    def _track_proba(self, track, X_selected, cache, name, indices=None):
        """Class-aligned probabilities for one track, memoised on the full batch."""
        if name in cache:
            proba = cache[name]
        else:
            proba = track.predict_proba(X_selected)
            cache[name] = proba
        track_classes = getattr(track.classifier, 'classes_', self.classes_)
        aligned = self._align_proba(proba, track_classes)
        return aligned if indices is None else aligned[indices]

    def _track_values(self, track, X_selected, cache, name):
        """Point predictions for one regression track, memoised on the full batch."""
        if name not in cache:
            cache[name] = np.asarray(track.predict(X_selected), dtype=float)
        return cache[name]

    def _fuse(self, X_selected: np.ndarray) -> Dict[str, Any]:
        """Route and fuse once. Returns the fused output plus the routing metadata.

        For classification the fused output is a normalized probability matrix; the label is
        derived from it downstream rather than produced independently. For regression it is a
        vector of point predictions.
        """
        n_samples = len(X_selected)
        active_mode = self._active_combination_mode_ or self.combination_mode
        is_clf = self.task_type == "classification"
        n_classes = len(self.classes_) if is_clf else 0
        cache: Dict[str, Any] = {}

        out: Dict[str, Any] = {'active_mode': active_mode, 'current_tracks': None,
                               'probabilities': None, 'values': None,
                               'routing_confidences': None}

        # ---------------------------------------------------------------- stacking path
        if active_mode == "stacking" and self.meta_learner_ is not None:
            stacked = self._stacking_predict_input(X_selected)
            if is_clf:
                probabilities = np.zeros((n_samples, n_classes))
                try:
                    meta_proba = self.meta_learner_.predict_proba(stacked)
                    meta_classes = getattr(self.meta_learner_, 'classes_', self.classes_)
                    probabilities = self._align_proba(meta_proba, meta_classes)
                except Exception as e:
                    # The meta-learner exposes no usable predict_proba. Fall back to its hard
                    # labels expressed as one-hot rows, so predict() keeps the labels it would
                    # have produced before and predict_proba() stays consistent with them —
                    # rather than the previous uniform-probability fallback, which silently
                    # made predict_proba disagree with predict on every sample.
                    logger.warning(f"stacking meta-learner predict_proba failed: {e}. "
                                   f"Falling back to one-hot rows from meta_learner.predict().")
                    try:
                        labels = np.asarray(self.meta_learner_.predict(stacked))
                        for j, cls in enumerate(self.classes_):
                            probabilities[labels == cls, j] = 1.0
                        empty = probabilities.sum(axis=1) == 0
                        probabilities[empty] = 1.0 / n_classes
                    except Exception:
                        probabilities[:] = 1.0 / n_classes
                out['probabilities'] = probabilities
                out['routing_confidences'] = np.max(probabilities, axis=1)
            else:
                out['values'] = np.asarray(self.meta_learner_.predict(stacked), dtype=float)
                # A Ridge meta-learner has no natural per-sample confidence; treated as fully
                # confident so routing-mode-only fallback/abstention remain no-ops here.
                out['routing_confidences'] = np.ones(n_samples)

            # PHASE7: difficulty-aware hedging. Blend the meta-learner's decision toward a flat
            # average over the SAME per-track outputs already sitting in `stacked` (no extra
            # track calls) in proportion to predicted difficulty -- see difficulty_aware_fusion's
            # docstring in __init__. Silently a no-op (falls through unblended) if the difficulty
            # model isn't available or the blend itself fails for any reason.
            if self.difficulty_aware_fusion and self.difficulty_model_ is not None:
                try:
                    difficulty = self._difficulty_from_selected(X_selected)
                    stack_track_names = self._stacking_track_names_ or []
                    n_tr = len(stack_track_names)
                    if n_tr > 0:
                        if is_clf:
                            track_block = stacked[:, :n_tr * n_classes].reshape(n_samples, n_tr, n_classes)
                            flat_avg = track_block.mean(axis=1)
                            row_sums = flat_avg.sum(axis=1, keepdims=True)
                            row_sums[row_sums == 0] = 1.0
                            flat_avg = flat_avg / row_sums
                            blend = difficulty[:, None]
                            blended = (1 - blend) * out['probabilities'] + blend * flat_avg
                            row_sums2 = blended.sum(axis=1, keepdims=True)
                            row_sums2[row_sums2 == 0] = 1.0
                            out['probabilities'] = blended / row_sums2
                            out['routing_confidences'] = np.max(out['probabilities'], axis=1)
                        else:
                            flat_avg = stacked[:, :n_tr].mean(axis=1)
                            out['values'] = (1 - difficulty) * out['values'] + difficulty * flat_avg
                except Exception as e:
                    logger.debug(f"PHASE7: difficulty-aware fusion blend failed (stacking path): {e}")

            # PHASE3: stacking evaluates EVERY expert for every sample — the meta-learner's
            # influence per expert is a global learned weight, not a per-sample weight vector,
            # so the honest fusion-width figure is the full expert count. Recorded so that
            # mean_effective_experts is defined for every fusion mode rather than None for this
            # one, which previously made the diagnostic unusable as a cost comparison.
            n_stack = len([t for t in self.tracks.keys() if t != 'correction_track'])
            if n_stack:
                self._record_expert_usage(np.full((n_samples, n_stack), 1.0 / n_stack))
            return out

        # ------------------------------------------------------- PHASE3-G2: flat average
        if active_mode == "flat_average":
            names = [t for t in self.tracks.keys() if t != 'correction_track']
            if is_clf:
                probabilities = np.zeros((n_samples, n_classes))
                used = 0
                for name in names:
                    track = self.tracks[name]
                    track.usage_count += n_samples
                    track.last_used = time.time()
                    probabilities += self._track_proba(track, X_selected, cache, name)
                    used += 1
                probabilities /= max(used, 1)
                row_sums = probabilities.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                out['probabilities'] = probabilities / row_sums
                out['routing_confidences'] = np.max(out['probabilities'], axis=1)
            else:
                acc = np.zeros(n_samples)
                used = 0
                for name in names:
                    track = self.tracks[name]
                    track.usage_count += n_samples
                    track.last_used = time.time()
                    acc += self._track_values(track, X_selected, cache, name)
                    used += 1
                out['values'] = acc / max(used, 1)
                out['routing_confidences'] = np.ones(n_samples)
            # Every expert contributes equally, so the realized fusion width is exactly the
            # number of experts — recorded through the same diagnostic as every other path.
            if names:
                self._record_expert_usage(
                    np.full((n_samples, len(names)), 1.0 / len(names)))
            return out

        # ---------------------------------------------------------------- routing paths
        track_weights = None
        router_weights = None
        current_tracks = None

        if (getattr(self, 'intelligent_routing', False)
                and getattr(self, 'global_router_', None) is not None
                and getattr(self, '_regional_routers_', {})):
            route_info = self._route_batch(X_selected, task_type=self.task_type,
                                           track_output_cache=cache)
            track_weights = route_info['track_weights']
            routing_confidences = route_info['routing_confidences']
            fallback_name = list(self.tracks.keys())[0]
            current_tracks = np.array([
                self.router_track_names_[idx] if idx < len(self.router_track_names_)
                else fallback_name
                for idx in route_info['chosen_track_indices']
            ], dtype=object)
        elif len(self.tracks) == 1 or not hasattr(self, 'router_') or self.router_ is None:
            track_name = list(self.tracks.keys())[0]
            current_tracks = np.full(n_samples, track_name, dtype=object)
            routing_confidences = np.ones(n_samples)
        else:
            route_info = self._route_with_router(X_selected, track_output_cache=cache)
            router_weights = route_info['router_weights']
            routing_confidences = route_info['routing_confidences']
            current_tracks = route_info['current_tracks']

        # PHASE2-D2: one weight source, one soft/hard decision. `weights` is whichever weight
        # matrix the active router produced; soft mode fuses over it, hard mode dispatches to
        # the argmax track. The intelligent-routing branch is no longer special-cased into
        # computing a fusion that a later branch throws away.
        weights = track_weights if track_weights is not None else router_weights

        # PHASE7: difficulty-aware hedging for soft routing -- blend the router's/track_weights'
        # per-sample weight vector toward a uniform vote over the same tracks, in proportion to
        # predicted difficulty, BEFORE it's used to fuse track outputs below. Applied at the
        # weight level (not by fusing twice) so it costs one extra difficulty prediction per
        # batch, not a second pass over every track. Only affects soft routing: hard routing's
        # dispatch decision comes from `current_tracks`/argmax, not from `weights`, and is left
        # unchanged (routing_mode="soft" is the default and what this project's benchmark uses).
        if (self.difficulty_aware_fusion and self.difficulty_model_ is not None
                and weights is not None and self.routing_mode == "soft"):
            try:
                difficulty = self._difficulty_from_selected(X_selected)
                uniform = np.full_like(weights, 1.0 / weights.shape[1]) if weights.shape[1] > 0 else weights
                blended = weights * (1 - difficulty)[:, None] + uniform * difficulty[:, None]
                row_sums = blended.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                weights = blended / row_sums
            except Exception as e:
                logger.debug(f"PHASE7: difficulty-aware fusion blend failed (routing path): {e}")

        use_soft = (self.routing_mode == "soft") and (weights is not None)

        if use_soft:
            self._record_expert_usage(weights)  # PHASE0 (measurement)
            if is_clf:
                probabilities = np.zeros((n_samples, n_classes))
                for i, tr_name in enumerate(self.router_track_names_):
                    if tr_name not in self.tracks:
                        continue
                    track = self.tracks[tr_name]
                    track.usage_count += n_samples
                    track.last_used = time.time()
                    probabilities += self._track_proba(track, X_selected, cache, tr_name) \
                        * weights[:, i:i + 1]
                row_sums = probabilities.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                out['probabilities'] = probabilities / row_sums
            else:
                stacked_preds = np.zeros((n_samples, len(self.router_track_names_)))
                for i, tr_name in enumerate(self.router_track_names_):
                    if tr_name not in self.tracks:
                        continue
                    track = self.tracks[tr_name]
                    track.usage_count += n_samples
                    track.last_used = time.time()
                    stacked_preds[:, i] = self._track_values(track, X_selected, cache, tr_name)
                out['values'] = (stacked_preds * weights).sum(axis=1)
        else:
            self._record_expert_usage(n_samples=n_samples)  # PHASE0 (measurement)
            if is_clf:
                probabilities = np.zeros((n_samples, n_classes))
            else:
                values = np.zeros(n_samples)
            available = [t for t in self.tracks.keys() if t != 'correction_track']
            for track_name in np.unique(current_tracks):
                indices = np.where(current_tracks == track_name)[0]
                resolved = track_name if track_name in self.tracks else (
                    available[0] if available else None)
                if resolved is None:
                    raise ValueError("No tracks available for prediction")
                track = self.tracks[resolved]
                track.usage_count += len(indices)
                track.last_used = time.time()
                if is_clf:
                    probabilities[indices] = self._track_proba(track, X_selected, cache,
                                                               resolved, indices)
                else:
                    values[indices] = self._track_values(track, X_selected, cache,
                                                          resolved)[indices]
            if is_clf:
                out['probabilities'] = probabilities
            else:
                out['values'] = values

        out['routing_confidences'] = routing_confidences
        out['current_tracks'] = current_tracks
        return out

    def _apply_correction(self, X_selected: np.ndarray, fused: Dict[str, Any],
                          count: bool) -> None:
        """Apply the residual correction track in place, in output space.

        Classification overrides the whole probability row (not just the collapsed label) for
        samples the router is unsure about and the correction track is confidently sure about;
        regression adds the predicted residual. `count` gates the diagnostic counter so that a
        predict()/predict_proba() pair over the same batch does not count twice.
        """
        if self.correction_track_ is None:
            return
        try:
            if self.task_type == "regression":
                fused['values'] = fused['values'] + self.correction_track_.predict(X_selected)
                if count:
                    self._correction_applied_count_ += len(X_selected)
                return

            # PHASE3-G1: gate on PREDICTIVE RISK using the threshold validated at fit time,
            # not on router confidence. Router confidence answers "which expert", which is a
            # different question from "is this prediction wrong" — on regime_clf only 2% of
            # samples ever fell below the old hard-coded 0.5, and those had to clear two further
            # conditions, so the correction track never fired in practice.
            if self.correction_risk_threshold_ is None:
                return
            risk = 1.0 - np.max(fused['probabilities'], axis=1)
            uncertain_mask = risk >= self.correction_risk_threshold_
            if uncertain_mask.sum() == 0:
                return
            X_uncertain = X_selected[uncertain_mask]
            correction_raw = self.correction_track_.predict_proba(X_uncertain)
            correction_conf = np.max(correction_raw, axis=1)
            # Same rule that was simulated during validation: selected by risk, and the
            # correction must itself be confident. The old extra clause comparing against
            # routing_confidences is dropped because it was not part of what is now validated,
            # and mixing an unvalidated condition into a validated rule makes the measured gain
            # inapplicable to what actually runs.
            override_mask = correction_conf >= self.correction_confidence_threshold
            final_idx = np.where(uncertain_mask)[0][override_mask]
            if len(final_idx) == 0:
                return
            correction_classes = getattr(self.correction_track_.classifier, 'classes_',
                                         self.classes_)
            aligned = self._align_proba(correction_raw, correction_classes)
            fused['probabilities'][final_idx] = aligned[override_mask]
            if count:
                self._correction_applied_count_ += len(final_idx)
            logger.debug(f"correction track overrode {len(final_idx)} samples")
        except Exception as e:
            logger.debug(f"Correction track skipped: {e}")

    def _apply_router_fallback(self, X_selected: np.ndarray, fused: Dict[str, Any]) -> None:
        """ACC-FIX 7: blend very low-confidence rows toward the all-track average, in place.

        PHASE2-D3: operates on the fused distribution for classification rather than on a
        one-hot vector rebuilt from an already-collapsed label, which is what predict() used to
        do. Blending a one-hot with an average discards the router's actual output and made the
        two entry points disagree on exactly the samples this safety net exists to protect.
        """
        if fused['active_mode'] != "routing" or self.router_fallback_threshold <= 0.0:
            return
        low_conf_mask = fused['routing_confidences'] < self.router_fallback_threshold
        if not np.any(low_conf_mask):
            return
        names = [t for t in self.tracks.keys() if t != 'correction_track']
        if not names:
            return
        idx = np.where(low_conf_mask)[0]
        blend = self.router_fallback_blend
        cache: Dict[str, Any] = {}
        n_ok = 0
        if self.task_type == "classification":
            avg = np.zeros((len(idx), len(self.classes_)))
            for name in names:
                try:
                    avg += self._track_proba(self.tracks[name], X_selected[idx], cache, name)
                    n_ok += 1
                except Exception:
                    pass
            if n_ok:
                avg /= n_ok
                fused['probabilities'][idx] = (blend * avg
                                               + (1 - blend) * fused['probabilities'][idx])
        else:
            avg = np.zeros(len(idx))
            for name in names:
                try:
                    avg += np.asarray(self.tracks[name].predict(X_selected[idx]), dtype=float)
                    n_ok += 1
                except Exception:
                    pass
            if n_ok:
                avg /= n_ok
                fused['values'][idx] = blend * avg + (1 - blend) * fused['values'][idx]
        if n_ok:
            logger.debug(f"ACC-FIX 7: blended {len(idx)} low-confidence rows "
                         f"(routing_confidence < {self.router_fallback_threshold}) "
                         f"toward the all-track average")

    def _predict_pipeline(self, X, count_diagnostics: bool) -> Dict[str, Any]:
        """Run the full shared inference pipeline and return the fused state.

        Order is fixed and identical for both entry points: route/fuse -> correction ->
        low-confidence fallback -> calibration. Abstention is applied afterwards by each entry
        point, because it takes a different form in label space and probability space.
        """
        X_selected = self._prepare_features(X)
        fused = self._fuse(X_selected)
        self._apply_correction(X_selected, fused, count=count_diagnostics)
        self._apply_router_fallback(X_selected, fused)
        if self.task_type == "classification":
            # PHASE2-D1: calibration is applied BEFORE the label is derived, so predict() and
            # predict_proba() argmax the same distribution. Previously only predict_proba()
            # calibrated, so a sigmoid/isotonic calibrator (which is fitted per class and does
            # not preserve argmax, unlike temperature scaling) could flip the label returned by
            # one method without flipping the other.
            fused['probabilities'] = self._apply_output_calibration(fused['probabilities'])
        fused['X_selected'] = X_selected
        return fused

    def predict(self, X) -> np.ndarray:
        """Predict labels (classification) or values (regression) via MoE routing.

        For classification this is exactly `classes_[argmax(predict_proba(X))]` by
        construction: both methods run the same fusion pipeline and argmax the same calibrated
        distribution.
        """
        check_is_fitted(self)
        _predict_start_time = time.time()  # PHASE6 (item 27): inference latency diagnostic

        fused = self._predict_pipeline(X, count_diagnostics=True)
        X_selected = fused['X_selected']
        n_samples = len(X_selected)
        routing_confidences = fused['routing_confidences']

        if self.task_type == "classification":
            predictions = self.classes_[np.argmax(fused['probabilities'], axis=1)]
        else:
            predictions = np.asarray(fused['values'], dtype=float)

        # Confidence-based abstention (label space).
        if self.abstention_threshold > 0.0:
            abstain_mask = self._compute_abstain_mask(X_selected, routing_confidences)
            if np.any(abstain_mask):
                self._abstention_count_ += int(np.sum(abstain_mask))  # PHASE6 item 27
                if self.task_type == "classification":
                    abs_val = self._get_abstention_sentinel()
                else:
                    abs_val = (self.abstention_class if self.abstention_class is not None
                               else np.nan)
                predictions = self._assign_with_dtype_promotion(predictions, abstain_mask,
                                                                abs_val)

        # PHASE4 (spec section 11 / items 33-E / rules 17-18): architecture mutation during an
        # ordinary predict() requires BOTH enable_dynamic_spawning and self_training, since with
        # no true labels here any spawned specialist would train on the model's own predictions.
        # Both default to False, so by default predict() never mutates the model.
        if self.enable_dynamic_spawning and self.self_training and self.fitted_:
            self._call_step_ += 1
            self._check_spawn_new_track(X_selected, predictions, routing_confidences)

        # Periodic track pruning
        self.prediction_count_ += n_samples
        if (self.enable_track_pruning and
                self.prediction_count_ >= getattr(self, '_last_prune_count', 0)
                + self.pruning_interval):
            self._prune_unused_tracks()
            self._last_prune_count = self.prediction_count_

        # PHASE6 (item 27): routing/latency history for get_routing_report(). Stacking mode has
        # no single chosen track per sample, so current_tracks is None and nothing is logged —
        # routing distribution/entropy are routing-mode-only diagnostics.
        if fused['current_tracks'] is not None:
            self._routing_history_.extend(np.asarray(fused['current_tracks']).tolist())
        self._inference_latencies_.append(time.time() - _predict_start_time)

        return predictions

    def predict_proba(self, X) -> np.ndarray:
        """Predict calibrated class probabilities (classification only).

        Runs the identical pipeline as predict(); predict() is argmax over this output.
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        check_is_fitted(self)

        fused = self._predict_pipeline(X, count_diagnostics=False)
        probabilities = fused['probabilities']

        # Abstention (probability space). Mirrors predict()'s label-space abstention on the
        # same mask, but expressed as a probability row rather than a sentinel label.
        if self.abstention_threshold > 0.0:
            abstain_mask = self._compute_abstain_mask(fused['X_selected'],
                                                      fused['routing_confidences'])
            n_classes = probabilities.shape[1]
            if (self.abstention_class is not None
                    and isinstance(self.abstention_class, int)
                    and 0 <= self.abstention_class < n_classes):
                probabilities[abstain_mask] = 0.0
                probabilities[abstain_mask, self.abstention_class] = 1.0
            else:
                probabilities[abstain_mask] = 0.0

        return probabilities
    
    def _check_spawn_new_track(self, X_selected: np.ndarray, predictions: np.ndarray, routing_confidences: np.ndarray,
                                spawn_confidence_labels: Optional[np.ndarray] = None):
        """PHASE4 (spec section 11): safe dynamic specialist spawning.

        Redesigned from the original "spawn immediately, self-label if no ground truth" scheme
        into the buffer-then-validate pipeline spec item 11 calls for:

            uncertain samples -> buffer -> wait for true labels -> accumulate enough examples
            -> evaluate whether a specialist is justified -> train -> validate -> integrate

        - When `spawn_confidence_labels` (true labels) are supplied — the partial_fit() path,
          which always has real labels for its chunk — uncertain rows are appended to a
          persistent labeled buffer (self._uncertain_buffer_X_/_y_). Nothing is trained on any
          single call; a specialist is only considered once the buffer holds at least
          self.min_spawn_samples rows AND self.spawn_cooldown calls have passed since the last
          spawn. When considered, the candidate is fit on a portion of the buffer and VALIDATED
          on a held-out portion (self.specialist_validation_fraction) — it's only accepted if
          it beats a min_spawn_gain threshold against the buffer's held-out portion. This
          mirrors _add_correction_track's OOF-gated pattern (Phase 1) rather than trusting a fit
          that "succeeded" as automatically useful.
        - When no true labels are available (the predict()-time path) this method is only
          reached at all when self.self_training=True was explicitly set (see predict()'s call
          site) — an intentionally separate, clearly-labeled-risky opt-in (spec item 11: "If
          self-training is supported, clearly mark it as optional and risky"). In that case the
          model's own predictions stand in for labels, still subject to the same
          min_spawn_samples/cooldown/validation gate — it just can't be leak-free.

        On acceptance, the new track is fully integrated (spec item 12): added to self.tracks,
        then refresh_router() (or refresh_fusion() in stacking mode) is called immediately so
        the router/meta-learner actually consumes it, and refresh_experts() recomputes
        diagnostics — no "spawned but silently unused" state.
        """
        if routing_confidences is None or len(routing_confidences) == 0:
            return
        if len(self.tracks) - int(self.correction_track_ is not None) >= self.max_dynamic_tracks + self.n_tracks:
            return  # PHASE4: respect max_dynamic_tracks as a real cap, not just a name

        is_self_training = spawn_confidence_labels is None
        low_confidence_mask = routing_confidences < 0.5
        if not is_self_training:
            # PHASE4 refinement: routing_confidence measures the router's confidence in WHICH
            # track it picked — not whether that track is actually right. A router can be
            # (and empirically is, on distribution-shifted data) confidently wrong: high
            # routing_confidence with low real accuracy. Since partial_fit always has true
            # labels, also flag samples the model got outright wrong, not just ones it was
            # unsure how to route — a strictly stronger signal for "this region needs a
            # specialist" than routing confidence alone.
            wrong_mask = np.asarray(predictions) != np.asarray(spawn_confidence_labels)
            low_confidence_mask = low_confidence_mask | wrong_mask
        uncertain_ratio = low_confidence_mask.sum() / len(routing_confidences)
        if uncertain_ratio <= self.confidence_spawn_threshold:
            return

        X_uncertain = X_selected[low_confidence_mask]
        if is_self_training:
            y_uncertain = predictions[low_confidence_mask]
            logger.warning(f"PHASE4: self_training=True — {uncertain_ratio:.1%} of predictions "
                            f"uncertain; buffering {len(X_uncertain)} samples using the model's "
                            f"OWN predictions as labels (risky: this can reinforce existing "
                            f"errors rather than correct them).")
        else:
            y_uncertain = np.asarray(spawn_confidence_labels)[low_confidence_mask]

        # Accumulate into the persistent buffer (bounded FIFO so it can't grow unboundedly
        # across a long streaming session).
        max_buffer = max(500, self.min_spawn_samples * 10)
        if self._uncertain_buffer_X_ is None:
            self._uncertain_buffer_X_ = X_uncertain
            self._uncertain_buffer_y_ = y_uncertain
        else:
            self._uncertain_buffer_X_ = np.concatenate([self._uncertain_buffer_X_, X_uncertain], axis=0)
            self._uncertain_buffer_y_ = np.concatenate([self._uncertain_buffer_y_, y_uncertain], axis=0)
        if len(self._uncertain_buffer_X_) > max_buffer:
            self._uncertain_buffer_X_ = self._uncertain_buffer_X_[-max_buffer:]
            self._uncertain_buffer_y_ = self._uncertain_buffer_y_[-max_buffer:]

        buffer_size = len(self._uncertain_buffer_X_)
        cooldown_ok = (self._call_step_ - self._last_spawn_step_) >= self.spawn_cooldown
        if buffer_size < self.min_spawn_samples or not cooldown_ok:
            logger.debug(f"PHASE4: spawn buffer at {buffer_size}/{self.min_spawn_samples} samples, "
                         f"cooldown_ok={cooldown_ok} — not yet evaluating a specialist.")
            return

        try:
            Xb, yb = self._uncertain_buffer_X_, self._uncertain_buffer_y_
            val_frac = float(np.clip(self.specialist_validation_fraction, 0.1, 0.5))
            try:
                fit_idx, val_idx = train_test_split(
                    np.arange(buffer_size), test_size=val_frac, random_state=self.random_state,
                    stratify=yb if self.task_type == "classification" and len(np.unique(yb)) > 1 else None
                )
            except ValueError:
                fit_idx, val_idx = np.arange(buffer_size), np.array([], dtype=int)

            n_existing = len(self.tracks)
            models = self._create_heterogeneous_models(n_samples=len(fit_idx))
            probe_clf = self._clone_track_model(models[n_existing % len(models)])
            self._fit_track_model(probe_clf, Xb[fit_idx], yb[fit_idx])

            # Validate: does the candidate specialist beat the CURRENT full ensemble on the
            # held-out portion of the very region it's meant to specialize in?
            if len(val_idx) == 0:
                # PHASE4-FIX: can't form a genuine held-out validation split (buffer too small,
                # or single-class making stratification impossible) — this validation gate is
                # the whole safety mechanism (spec item 11: "validate specialist" is a real
                # step, not a formality), so the conservative choice is to NOT spawn yet rather
                # than silently accept. An earlier version defaulted `gain = min_spawn_gain`
                # here, which trivially passed the `gain < threshold` check regardless of how
                # strict the threshold was set — defeating the gate entirely in this edge case.
                # Keep buffering; a larger buffer will eventually split successfully.
                logger.debug("PHASE4: buffer too small/homogeneous to form a validation split; "
                             "continuing to buffer rather than spawning unvalidated.")
                return
            candidate_preds = probe_clf.predict(Xb[val_idx])
            if self.task_type == "classification":
                candidate_score = float(np.mean(candidate_preds == yb[val_idx]))
                baseline_preds = self.predict(Xb[val_idx])
                baseline_score = float(np.mean(baseline_preds == yb[val_idx]))
                gain = candidate_score - baseline_score
            else:
                candidate_mae = float(np.mean(np.abs(yb[val_idx] - candidate_preds)))
                baseline_preds = self.predict(Xb[val_idx])
                baseline_mae = float(np.mean(np.abs(yb[val_idx] - baseline_preds)))
                gain = (baseline_mae - candidate_mae) / max(baseline_mae, 1e-8)

            # PHASE4 (spec item 24): under detected drift, specialist creation is deliberately
            # made easier — a stale ensemble's "baseline" comparison is itself less trustworthy.
            effective_gain_threshold = self.min_spawn_gain / 2.0 if self.drift_detected_ else self.min_spawn_gain

            if gain < effective_gain_threshold:
                logger.info(f"PHASE4: candidate specialist did not clear the validation gate "
                            f"(gain={gain:.4f} < {effective_gain_threshold:.4f}); discarding and "
                            f"continuing to buffer.")
                self._last_spawn_step_ = self._call_step_  # still respect cooldown before retrying
                return

            # Passed — refit on the FULL buffer for deployment and integrate it properly.
            final_clf = self._clone_track_model(models[n_existing % len(models)])
            self._fit_track_model(final_clf, Xb, yb)
            new_track_name = f"track_spawn_{n_existing}_{self._dynamic_tracks_created}"
            self.tracks[new_track_name] = Track(new_track_name, final_clf)
            self._dynamic_tracks_created += 1
            self._last_spawn_step_ = self._call_step_
            self._uncertain_buffer_X_ = None
            self._uncertain_buffer_y_ = None

            logger.info(f"PHASE4: accepted new specialist '{new_track_name}' "
                        f"(validated gain={gain:.4f}, trained on {buffer_size} buffered samples"
                        f"{', self-trained labels' if is_self_training else ', true labels'}).")

            # spec item 12: full integration, not just "sits in self.tracks unused".
            if self._active_combination_mode_ == "stacking":
                self.refresh_fusion()
            else:
                self.refresh_router()
            self.refresh_experts()
        except Exception as e:
            logger.debug(f"PHASE4: specialist spawning evaluation failed: {e}")

    def _expected_calibration_error(self, proba: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> float:
        """Standard reliability-diagram Expected Calibration Error: bin predictions by their
        top-class confidence, and average |accuracy - confidence| within each bin, weighted by
        bin size. Lower is better; 0 = perfectly calibrated on this sample."""
        confidences = proba.max(axis=1)
        predictions = np.argmax(proba, axis=1)
        correct = (predictions == y_idx).astype(float)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(y_idx)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
            if not np.any(in_bin):
                continue
            bin_acc = correct[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)

    def _apply_output_calibration(self, proba: np.ndarray) -> np.ndarray:
        """Apply the calibration transform selected by _fit_output_calibration, if any."""
        if self._calibration_method_ in (None, "none") or self._calibration_params_ is None:
            return proba
        try:
            if self._calibration_method_ == "temperature":
                T = self._calibration_params_["T"]
                logits = np.log(np.clip(proba, 1e-12, 1.0)) / T
                exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                return exp_logits / exp_logits.sum(axis=1, keepdims=True)
            elif self._calibration_method_ == "sigmoid":
                models = self._calibration_params_["models"]
                calibrated = np.column_stack([m.predict_proba(proba[:, c].reshape(-1, 1))[:, 1] for c, m in enumerate(models)])
                row_sums = calibrated.sum(axis=1, keepdims=True)
                row_sums[row_sums <= 0] = 1.0
                return calibrated / row_sums
            elif self._calibration_method_ == "isotonic":
                models = self._calibration_params_["models"]
                calibrated = np.column_stack([m.predict(proba[:, c]) for c, m in enumerate(models)])
                row_sums = calibrated.sum(axis=1, keepdims=True)
                row_sums[row_sums <= 0] = 1.0
                return calibrated / row_sums
        except Exception as e:
            logger.debug(f"PHASE5: calibration application failed, returning uncalibrated probabilities: {e}")
        return proba

    def _fit_output_calibration(self, X_calib_raw, y_calib_raw):
        """PHASE5 (spec item 21): fit + VALIDATE ensemble-level probability calibration.

        Tries {temperature, sigmoid, isotonic-if-enough-data} against a "none" baseline on a
        genuinely held-out split of the calibration slice, and keeps whichever has the lowest
        log loss on that held-out split. Because "none" is itself a candidate, this can never
        report calibration as an improvement unless it measurably is one — directly satisfying
        spec item 21's "do not claim calibration is successful without measuring it".
        """
        from sklearn.metrics import log_loss, brier_score_loss
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        try:
            # self._calibration_method_ is still None here, so this predict_proba() call
            # returns RAW (uncalibrated) fused probabilities.
            raw_proba_all = self.predict_proba(X_calib_raw)
            y_calib_arr = np.asarray(y_calib_raw)
            y_idx_all = np.array([np.where(self.classes_ == v)[0][0] if v in self.classes_ else 0 for v in y_calib_arr])

            n = len(y_idx_all)
            fit_idx, val_idx = train_test_split(
                np.arange(n), test_size=0.5, random_state=self.random_state,
                stratify=y_idx_all if len(np.unique(y_idx_all)) > 1 else None
            )
            raw_fit, raw_val = raw_proba_all[fit_idx], raw_proba_all[val_idx]
            y_fit, y_val = y_idx_all[fit_idx], y_idx_all[val_idx]
            n_classes = len(self.classes_)

            candidates = {}

            # "none" — the baseline every other method must beat.
            candidates["none"] = {"params": None, "proba_val": raw_val}

            # "temperature" — single scalar, minimizes NLL on the fit split via coarse-to-fine
            # grid search (cheap, robust, no extra dependency beyond numpy).
            def nll_at_T(T):
                logits = np.log(np.clip(raw_fit, 1e-12, 1.0)) / T
                exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                p = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                return log_loss(y_fit, p, labels=np.arange(n_classes))
            T_grid = np.geomspace(0.1, 10.0, 25)
            best_T = min(T_grid, key=nll_at_T)
            logits = np.log(np.clip(raw_val, 1e-12, 1.0)) / best_T
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            temp_proba_val = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            candidates["temperature"] = {"params": {"T": float(best_T)}, "proba_val": temp_proba_val}

            # "sigmoid" (Platt scaling, one-vs-rest) and "isotonic" — only attempted if there's
            # enough data per class to fit them without just memorizing noise.
            min_per_class = min((y_fit == c).sum() for c in range(n_classes)) if n_classes > 0 else 0
            if min_per_class >= 15:
                sigmoid_models, iso_models = [], []
                for c in range(n_classes):
                    yc = (y_fit == c).astype(int)
                    lr = LogisticRegression()
                    lr.fit(raw_fit[:, c].reshape(-1, 1), yc)
                    sigmoid_models.append(lr)
                    if min_per_class >= 40:
                        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                        iso.fit(raw_fit[:, c], yc)
                        iso_models.append(iso)

                sig_val = np.column_stack([m.predict_proba(raw_val[:, c].reshape(-1, 1))[:, 1] for c, m in enumerate(sigmoid_models)])
                sig_val = sig_val / np.clip(sig_val.sum(axis=1, keepdims=True), 1e-8, None)
                candidates["sigmoid"] = {"params": {"models": sigmoid_models}, "proba_val": sig_val}

                if len(iso_models) == n_classes:
                    iso_val = np.column_stack([m.predict(raw_val[:, c]) for c, m in enumerate(iso_models)])
                    iso_val = iso_val / np.clip(iso_val.sum(axis=1, keepdims=True), 1e-8, None)
                    candidates["isotonic"] = {"params": {"models": iso_models}, "proba_val": iso_val}

            # Score every candidate on the SAME held-out val split.
            scored = {}
            for name, c in candidates.items():
                p = np.clip(c["proba_val"], 1e-12, 1.0)
                ll = float(log_loss(y_val, p, labels=np.arange(n_classes)))
                onehot = np.eye(n_classes)[y_val]
                brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
                ece = self._expected_calibration_error(p, y_val)
                scored[name] = {"log_loss": ll, "brier": brier, "ece": ece}

            if self.calibration_method != "auto" and self.calibration_method in candidates:
                winner = self.calibration_method
            else:
                winner = min(scored, key=lambda k: scored[k]["log_loss"])

            self._calibration_method_ = winner
            self._calibration_params_ = candidates[winner]["params"]
            self.calibration_report_ = {
                "selected_method": winner,
                "candidates": scored,
                "pre_calibration": scored["none"],
                "post_calibration": scored[winner],
                "n_calibration_samples": n,
            }
            logger.info(f"PHASE5: output calibration selected '{winner}' "
                        f"(log loss {scored['none']['log_loss']:.4f} -> {scored[winner]['log_loss']:.4f}, "
                        f"Brier {scored['none']['brier']:.4f} -> {scored[winner]['brier']:.4f}, "
                        f"ECE {scored['none']['ece']:.4f} -> {scored[winner]['ece']:.4f}).")
        except Exception as e:
            logger.warning(f"PHASE5: output calibration fitting failed, predict_proba will "
                            f"remain uncalibrated: {e}")
            self._calibration_method_ = None
            self._calibration_params_ = None
            self.calibration_report_ = None

    def get_calibration_report(self) -> Dict[str, Any]:
        """PHASE5 (spec item 21): pre/post calibration metrics (log loss, Brier, ECE) for
        every candidate method tried, measured on a genuinely held-out calibration split.
        Raises if calibrate_output=False or calibration wasn't fitted successfully."""
        check_is_fitted(self)
        if self.calibration_report_ is None:
            raise ValueError("Calibration report not available (calibrate_output=False, or "
                              "calibration fitting failed/was skipped for this fit).")
        return self.calibration_report_

    def get_expert_report(self) -> List[Dict[str, Any]]:
        """PHASE6 (spec item 27): per-expert breakdown, one row per current track. Every field
        here comes from state that's actually tracked during normal operation — usage_count and
        prediction_times are updated on every real predict() call (see Track.predict/
        predict_proba), performance_score is the OOF leave-one-out marginal-contribution
        estimate from _compute_expert_diagnostics (Phase 3) — nothing here is fabricated after
        the fact.
        """
        check_is_fitted(self)
        diagnostics = self.diversity_diagnostics_
        report = []
        for name, track in self.tracks.items():
            redundancy = None
            if diagnostics is not None and name in diagnostics.get('pairwise_pred_correlation_by_name', {}):
                corr_row = diagnostics['pairwise_pred_correlation_by_name'][name]
                other = [v for k, v in corr_row.items() if k != name and k in self.tracks]
                redundancy = float(np.mean(other)) if other else None
            report.append({
                'name': name,
                'model_type': type(track.classifier).__name__,
                'is_dynamic': name.startswith('track_spawn_'),
                'is_correction_track': name == 'correction_track',
                'usage_count': track.usage_count,
                'capacity_violations': track.capacity_violations,
                'avg_prediction_latency_s': float(np.mean(track.prediction_times)) if track.prediction_times else None,
                'marginal_oof_contribution': track.performance_score,
                'mean_prediction_correlation_with_others': redundancy,
                'seconds_since_last_used': time.time() - track.last_used,
            })
        return report

    def _record_expert_usage(self, weights: Optional[np.ndarray] = None,
                             n_samples: Optional[int] = None) -> None:
        """PHASE0 (measurement): record the effective number of experts fused per sample.

        Called at the point where the output is actually produced, so the number reflects the
        realized fusion rather than a configured budget. `weights=None` means a single expert
        produced the output (hard routing / single-track fallback) and records exactly 1.0.
        Otherwise records the perplexity exp(H(w)) of the row-normalized fusion weights, which
        equals 1.0 for a one-hot weight vector and n for a uniform blend over n experts.
        Purely diagnostic: nothing in the prediction path reads this.
        """
        try:
            if weights is None:
                if not n_samples:
                    return
                self._effective_experts_history_.extend([1.0] * int(n_samples))
                return
            w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
            if w.ndim != 2 or w.size == 0:
                return
            row_sums = w.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            w = w / row_sums
            entropy = -np.sum(w * np.log(w + 1e-12), axis=1)
            self._effective_experts_history_.extend(np.exp(entropy).tolist())
        except Exception:
            # Diagnostics must never break inference.
            pass

    def reset_diagnostics(self) -> None:
        """PHASE0 (measurement): clear all rolling inference-time diagnostic counters.

        The counters behind get_diagnostics() (abstention count, correction applications,
        prediction count, routing history, latencies) are CUMULATIVE across every predict()
        call the model has ever served, including internal calls made during fit()-time
        calibration. Reporting `abstention_count / prediction_count` over that mixture answers
        a question nobody asked. Call this immediately before a evaluation pass to scope the
        counters to exactly that pass.

        Does not touch fitted state — the model remains fitted and unchanged.
        """
        self._abstention_count_ = 0
        self._correction_applied_count_ = 0
        self.prediction_count_ = 0
        self._routing_history_.clear()
        self._inference_latencies_.clear()
        self._effective_experts_history_.clear()
        self._dynamic_top_k_history_.clear()

    def get_routing_report(self) -> Dict[str, Any]:
        """PHASE6 (spec item 27): routing distribution/entropy over recent predict() calls
        (routing-mode only — see predict()'s note on why stacking mode has nothing to log
        here), built from self._routing_history_, a real rolling log of which track was
        actually selected per sample, not a snapshot or an estimate.
        """
        check_is_fitted(self)
        if len(self._routing_history_) == 0:
            # PHASE3 fix: fusion-width diagnostics are meaningful even when there is no
            # per-sample "chosen track" to log (stacking and flat_average both fuse every
            # expert), so this early return must still carry them. Previously it returned a
            # dict missing every effective-expert key, so any caller that read them —
            # get_diagnostics() and the benchmark among them — raised KeyError the moment a
            # non-routing fusion mode was used.
            report = {
                'note': "No per-sample routing history recorded — either no predict() calls "
                        "have been made since fit(), or the active fusion mode (stacking / "
                        "flat_average) has no single 'chosen track' per sample to log.",
                'routing_distribution': {},
                'routing_entropy': None,
                'routing_entropy_normalized': None,
                'n_samples_logged': 0,
                'n_distinct_tracks_used': 0,
            }
            self._attach_fusion_width_diagnostics(report)
            return report
        names, counts = np.unique(list(self._routing_history_), return_counts=True)
        total = counts.sum()
        probs = counts / total
        # PHASE0: the +1e-12 smoothing made a degenerate single-track history report a
        # NEGATIVE entropy (-1e-12) instead of 0. Clamp at 0 — entropy is non-negative.
        entropy = max(0.0, float(-np.sum(probs * np.log(probs + 1e-12))))
        max_entropy = float(np.log(len(names))) if len(names) > 1 else 1.0
        report = {
            'routing_distribution': {str(n): int(c) for n, c in zip(names, counts)},
            'routing_entropy': entropy,
            'routing_entropy_normalized': entropy / max_entropy if max_entropy > 0 else 0.0,
            'n_samples_logged': int(total),
            'n_distinct_tracks_used': int(len(names)),
        }
        # PHASE0: realized fusion width. `mean_effective_experts` is the honest answer to
        # "how many experts does a prediction actually cost" — the adaptive top-k budget is
        # reported alongside it but is a different quantity (a budget, not a realization).
        self._attach_fusion_width_diagnostics(report)
        return report

    def _attach_fusion_width_diagnostics(self, report: Dict[str, Any]) -> None:
        """Add realized-fusion-width diagnostics to a routing report, in place.

        Kept separate from the routing-distribution block so that both the normal path and the
        no-routing-history path (stacking, flat_average) report the same keys.
        """
        if len(self._effective_experts_history_) > 0:
            eff = np.asarray(self._effective_experts_history_, dtype=float)
            report['mean_effective_experts'] = float(eff.mean())
            report['median_effective_experts'] = float(np.median(eff))
            report['max_effective_experts'] = float(eff.max())
            report['n_effective_experts_logged'] = int(eff.size)
        else:
            report['mean_effective_experts'] = None
            report['median_effective_experts'] = None
            report['max_effective_experts'] = None
            report['n_effective_experts_logged'] = 0
        if len(self._dynamic_top_k_history_) > 0:
            ks, kc = np.unique(np.asarray(self._dynamic_top_k_history_, dtype=int), return_counts=True)
            report['dynamic_top_k_distribution'] = {int(k): int(c) for k, c in zip(ks, kc)}
            report['mean_dynamic_top_k'] = float(np.average(ks, weights=kc))
        else:
            report['dynamic_top_k_distribution'] = {}
            report['mean_dynamic_top_k'] = None

    def get_diagnostics(self) -> Dict[str, Any]:
        """PHASE6 (spec item 27): consolidated research-grade diagnostics snapshot. Aggregates
        get_expert_report()/get_routing_report() plus every other piece of state this
        implementation actually tracks — diversity/contribution diagnostics (Phase 3),
        calibration report (Phase 5), drift status (Phase 4), correction/abstention usage
        (Phase 6 counters), and inference latency. Fields that were never computed for this fit
        (e.g. calibration when calibrate_output=False) are reported as None rather than omitted
        or fabricated, so callers can tell "not measured" apart from "measured as zero/absent".
        """
        check_is_fitted(self)
        expert_report = self.get_expert_report()
        routing_report = self.get_routing_report()

        total_predictions = max(1, self.prediction_count_)
        return {
            'task_type': self.task_type,
            'n_features_in': self.n_features_in_,
            'active_combination_mode': self._active_combination_mode_,
            'combination_mode_scores': getattr(self, 'combination_mode_scores_', None),  # PHASE3-G2
            'correction_risk_threshold': getattr(self, 'correction_risk_threshold_', None),  # PHASE3-G1
            'correction_validated_gain': getattr(self, 'correction_gain_', None),  # PHASE3-G1
            'routing_mode': self.routing_mode,
            'intelligent_routing': bool(getattr(self, 'intelligent_routing', False)),
            'n_experts': len(self.tracks),
            'n_dynamic_experts_created': self._dynamic_tracks_created,
            'minimum_experts': self.minimum_experts,
            'expert_report': expert_report,
            'routing_report': routing_report,
            # PHASE0: promoted to the top level because spec item 36 requires "average experts
            # used per sample" as a headline benchmark column.
            'mean_effective_experts': routing_report.get('mean_effective_experts'),
            'mean_dynamic_top_k': routing_report.get('mean_dynamic_top_k'),
            'diversity_score': self.diversity_diagnostics_['diversity_score'] if self.diversity_diagnostics_ else None,
            'mean_pairwise_error_correlation': self.diversity_diagnostics_['mean_pairwise_error_correlation'] if self.diversity_diagnostics_ else None,
            'correction_track_active': self.correction_track_ is not None,
            'correction_gain': self.correction_gain_,
            'correction_applications': self._correction_applied_count_,
            'correction_application_rate': self._correction_applied_count_ / total_predictions,
            'abstention_count': self._abstention_count_,
            'abstention_rate': self._abstention_count_ / total_predictions,
            'calibration_report': self.calibration_report_,
            'conformal_calibrated': self._conformal_calibrated_,
            'drift_detected': self.drift_detected_,
            'drift_event_count': self._drift_event_count_,
            'prediction_count': self.prediction_count_,
            'mean_inference_latency_s': float(np.mean(self._inference_latencies_)) if self._inference_latencies_ else None,
            'p95_inference_latency_s': float(np.percentile(self._inference_latencies_, 95)) if len(self._inference_latencies_) >= 5 else None,
        }

    def explain(self, X, index: int = 0) -> Dict[str, Any]:
        """PHASE6 (spec item 28): explain a single prediction, tracing the ACTUAL decision path
        (not a separately-computed approximation of it) — the same routing helper, the same
        difficulty/signal computation, the same correction-track logic predict()/predict_proba()
        use, run once here and reported back rather than reimplemented for explanation purposes.

        Returns a dict describing, for row `index` of X:
            prediction, predicted_probabilities (classification only)
            selected_track(s) and their routing weights
            per-track predictions (what each expert individually would have said)
            expert_utility (predicted expected loss per track, if router_loss_aware was used)
            uncertainty (routing confidence for the resolved decision)
            difficulty (if a difficulty model was fitted)
            structural_signals (disagreement/entropy/density/cluster_distance/outlier)
            correction_applied / abstained (booleans, matching what predict() would actually do)
        """
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X_arr = check_array(X, accept_sparse=False)  # allow-nan is set automatically by the check_array wrapper above
        else:
            X_arr = X
        if index < 0 or index >= len(X_arr):
            raise ValueError(f"index {index} out of range for X with {len(X_arr)} rows")
        X_row = X_arr[index:index + 1] if not isinstance(X_arr, pd.DataFrame) else X_arr.iloc[index:index + 1]

        X_scaled = self.preprocessor_.transform(X_row)
        X_selected = self.feature_selector_.transform(X_scaled) if self.feature_selector_ is not None else X_scaled

        result: Dict[str, Any] = {'task_type': self.task_type}

        # Per-track individual predictions — what each expert would say on its own.
        track_names = [t for t in self.tracks.keys() if t != 'correction_track']
        per_track = {}
        per_track_class_idx = {}  # PHASE6: raw class index per track, for structural signals below
        for name in track_names:
            try:
                if self.task_type == "classification":
                    proba = self.tracks[name].predict_proba(X_selected)
                    track_classes = getattr(self.tracks[name].classifier, 'classes_', self.classes_)
                    proba_aligned = self._align_proba(proba, track_classes)
                    class_idx = int(np.argmax(proba_aligned[0]))
                    per_track_class_idx[name] = class_idx
                    per_track[name] = {
                        'predicted_class': str(self.classes_[class_idx]),
                        'confidence': float(np.max(proba_aligned[0])),
                    }
                else:
                    per_track[name] = {'prediction': float(self.tracks[name].predict(X_selected)[0])}
            except Exception as e:
                per_track[name] = {'error': str(e)}
        result['expert_predictions'] = per_track

        # Actual routing decision — reuses the SAME helper predict()/predict_proba() call.
        if self.router_ is not None and self._active_combination_mode_ != "stacking":
            route_info = self._route_with_router(X_selected)
            result['selected_track'] = str(route_info['current_tracks'][0])
            result['routing_weights'] = {
                self.router_track_names_[i]: float(route_info['router_weights'][0, i])
                for i in range(len(self.router_track_names_)) if i < route_info['router_weights'].shape[1]
            }
            result['routing_confidence'] = float(route_info['routing_confidences'][0])
        elif self._active_combination_mode_ == "stacking":
            result['selected_track'] = None
            result['routing_weights'] = None
            result['note'] = "Stacking mode: prediction is a learned combination of all experts, not a single routing choice."
        else:
            result['selected_track'] = track_names[0] if track_names else None
            result['routing_weights'] = None

        # Loss-aware expert utility, if that router was trained.
        if self.router_loss_aware and self.expert_loss_model_ is not None:
            try:
                predicted_loss = np.atleast_2d(self.expert_loss_model_.predict(X_selected))[0]
                result['expert_expected_loss'] = {
                    self._loss_aware_track_names_[i]: float(predicted_loss[i])
                    for i in range(min(len(self._loss_aware_track_names_), len(predicted_loss)))
                }
            except Exception:
                result['expert_expected_loss'] = None

        # Difficulty + structural signals — same source predict_difficulty() uses.
        if self.difficulty_model_ is not None:
            try:
                result['difficulty'] = float(self._difficulty_from_selected(X_selected)[0])
            except Exception:
                result['difficulty'] = None
        else:
            result['difficulty'] = None
        try:
            # Match _fit_difficulty_model / _difficulty_from_selected's convention: numeric
            # per-track predictions (class INDEX for classification, since raw labels may be
            # non-numeric strings) so the disagreement/entropy signals are actually informative
            # instead of silently defaulting to 0 with track_predictions=None.
            if self.task_type == "regression":
                track_pred_matrix = np.array([[v.get('prediction', np.nan) for v in per_track.values()]])
            else:
                track_pred_matrix = np.array([[
                    np.searchsorted(self.classes_, v['predicted_class']) if 'predicted_class' in v else np.nan
                    for v in per_track.values()
                ]])
            signals = self.signal_extractor_.extract_signals(X_selected, track_predictions=track_pred_matrix)
            result['structural_signals'] = {
                'disagreement': float(signals[0, 0]), 'entropy': float(signals[0, 1]),
                'density': float(signals[0, 2]), 'cluster_distance': float(signals[0, 3]),
                'outlier_score': float(signals[0, 4]),
            }
        except Exception:
            result['structural_signals'] = None

        # Final prediction, and whether correction/abstention would fire — via the real path.
        final_pred = self.predict(X_row)
        result['prediction'] = str(final_pred[0]) if self.task_type == "classification" else float(final_pred[0])
        if self.task_type == "classification":
            proba = self.predict_proba(X_row)
            result['predicted_probabilities'] = {str(c): float(p) for c, p in zip(self.classes_, proba[0])}
            result['abstained'] = bool(self.abstention_threshold > 0.0 and
                                        result['prediction'] == str(self._get_abstention_sentinel()))
        result['correction_track_available'] = self.correction_track_ is not None

        return result

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate enhanced model score."""
        y_pred = self.predict(X)
        if self.task_type == "classification":
            return accuracy_score(y, y_pred)
        else:
            return -mean_squared_error(y, y_pred)

    def _compute_ensemble_disagreement(self, X) -> np.ndarray:
        """PHASE1 item 7: per-sample std across production tracks' regression point
        predictions. This is an epistemic-uncertainty PROXY (not a Bayesian posterior) —
        it reflects how much the heterogeneous experts disagree given the input, which tends
        to be larger in regions the ensemble has less reliable knowledge of. Used both as a
        diagnostic and as the fallback predict_interval() width when no calibration data was
        available at fit time.
        """
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            X_arr = check_array(X, accept_sparse=False)  # allow-nan is set automatically by the check_array wrapper above
        else:
            X_arr = X
        X_scaled = self.preprocessor_.transform(X_arr)
        X_selected = self.feature_selector_.transform(X_scaled) if self.feature_selector_ is not None else X_scaled
        track_names = [t for t in self.tracks.keys() if t != 'correction_track']
        if not track_names:
            return np.zeros(len(X_selected))
        preds = np.full((len(X_selected), len(track_names)), np.nan)
        for i, name in enumerate(track_names):
            try:
                preds[:, i] = self.tracks[name].predict(X_selected)
            except Exception:
                pass
        with np.errstate(invalid='ignore'):
            disagreement = np.nanstd(preds, axis=1)
        return np.nan_to_num(disagreement, nan=0.0)

    def predict_interval(self, X, confidence: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PHASE1 item 7: regression prediction intervals — genuine, non-constant uncertainty.

        Returns (point_prediction, lower_bound, upper_bound).

        When a held-out conformal calibration slice was available at fit() time
        (self._conformal_calibrated_, controlled by enable_conformal/conformal_calibration_fraction/
        conformal_min_samples), this uses split conformal prediction: the interval half-width is
        the finite-sample-corrected (1 - alpha) quantile of |y - prediction| measured on data the
        model never trained on. Under the standard conformal-prediction exchangeability
        assumption this gives approximately valid marginal coverage; it is NOT a guarantee under
        concept drift or if the calibration set is small/unrepresentative.

        Otherwise it falls back to an UNCALIBRATED interval based on ensemble disagreement
        (std across heterogeneous tracks) scaled by a normal-approximation z-factor — this is
        explicitly a heuristic, not a validated coverage guarantee, and is logged as such.
        """
        if self.task_type != "regression":
            raise ValueError("predict_interval is only available for regression tasks")
        check_is_fitted(self)
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must be strictly between 0 and 1")

        point_pred = np.asarray(self.predict(X), dtype=float)

        if self._conformal_calibrated_ and self._conformal_residuals_ is not None and len(self._conformal_residuals_) > 0:
            residuals = self._conformal_residuals_
            n = len(residuals)
            alpha = 1.0 - confidence
            # IMPROVEMENT: Use the conformal order statistic; interpolation undercovers small sets.
            rank = int(np.ceil((n + 1) * (1 - alpha)))
            half_width = float(residuals[rank - 1]) if rank <= n else float('inf')
            lower = point_pred - half_width
            upper = point_pred + half_width
            logger.debug(f"predict_interval: split-conformal half-width={half_width:.4f} "
                         f"(calibrated on {n} held-out samples)")
        else:
            disagreement = self._compute_ensemble_disagreement(X)
            try:
                from scipy.stats import norm
                z = float(norm.ppf(0.5 + confidence / 2.0))
            except Exception:
                z = 1.645 if confidence <= 0.90 else 2.576
            half_width = z * disagreement
            lower = point_pred - half_width
            upper = point_pred + half_width
            logger.warning("predict_interval: no conformal calibration data was available at "
                            "fit time (dataset too small, or enable_conformal=False) — returning "
                            "an UNCALIBRATED ensemble-disagreement-based interval, not a "
                            "validated coverage guarantee.")

        return point_pred, lower, upper

    def save_model(self, filename: str):
        """PHASE6 item 29: save the trained model with research-grade metadata.

        The pickled model object itself remains the source of truth for actually running the
        model (predict/predict_proba/etc all just work off `self`, unchanged) — the metadata
        dict is an additional, human-and-machine-readable record of what was saved, so a saved
        file can be inspected/validated without unpickling the full model, and so load_model()
        can warn about architecture-version or schema mismatches.
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before saving")

        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except Exception:
            sklearn_version = None
        try:
            import joblib as _joblib_mod
            joblib_version = _joblib_mod.__version__
        except Exception:
            joblib_version = None

        selected_feature_indices = None
        if self.feature_selector_ is not None:
            try:
                selected_feature_indices = self.feature_selector_.get_support(indices=True).tolist()
            except Exception:
                selected_feature_indices = None

        save_data = {
            'model': self,
            'metadata': {
                # --- schema / provenance (item 29) ---
                'architecture_version': self._ARCHITECTURE_VERSION,
                'library_versions': {'numpy': np.__version__, 'sklearn': sklearn_version, 'joblib': joblib_version},
                'random_state': self.random_state,
                'save_timestamp': time.time(),
                # --- task / preprocessing schema ---
                'task_type': self.task_type,
                'n_features': self.n_features_in_,
                'feature_names': getattr(self, 'feature_names_in_', None),
                'selected_feature_indices': selected_feature_indices,
                'classes': self.classes_.tolist() if self.task_type == "classification" and self.classes_ is not None else None,
                # --- expert / router metadata ---
                'n_tracks': len(self.tracks),
                'expert_types': {name: type(t.classifier).__name__ for name, t in self.tracks.items()},
                'n_dynamic_experts_created': self._dynamic_tracks_created,
                'router_type': self.router_type,
                'routing_mode': getattr(self, 'routing_mode', 'hard'),
                'cluster_experts': getattr(self, 'cluster_experts', False),
                'pruning_enabled': self.enable_track_pruning,
                'minimum_experts': self.minimum_experts,
                'intelligent_routing': getattr(self, 'intelligent_routing', False),
                'routing_regions': getattr(self, 'routing_regions', None),
                'routing_bootstrap_ensemble_size': getattr(self, 'routing_bootstrap_ensemble_size', None),
                'routing_memory_size': getattr(self, 'routing_memory_size', None),
                'adaptive_top_k_max': getattr(self, 'adaptive_top_k_max', None),
                # --- fusion / routing configuration ---
                'active_combination_mode': self._active_combination_mode_,
            'combination_mode_scores': getattr(self, 'combination_mode_scores_', None),  # PHASE3-G2
            'correction_risk_threshold': getattr(self, 'correction_risk_threshold_', None),  # PHASE3-G1
            'correction_validated_gain': getattr(self, 'correction_gain_', None),  # PHASE3-G1
                'combination_mode_requested': self.combination_mode,
                'router_loss_aware': self.router_loss_aware,
                # --- calibration / correction / uncertainty state ---
                'conformal_calibrated': getattr(self, '_conformal_calibrated_', False),
                'output_calibration_method': self._calibration_method_,
                'calibration_report_summary': (
                    {'selected_method': self.calibration_report_['selected_method'],
                     'pre_log_loss': self.calibration_report_['pre_calibration']['log_loss'],
                     'post_log_loss': self.calibration_report_['post_calibration']['log_loss']}
                    if self.calibration_report_ is not None else None
                ),
                'correction_track_active': self.correction_track_ is not None,
                'correction_gain': getattr(self, 'correction_gain_', None),
                'difficulty_model_active': self.difficulty_model_ is not None,
                # --- drift / online-adaptation configuration ---
                'drift_threshold': self.drift_threshold,
                'drift_detected_at_save': self.drift_detected_,
                'drift_event_count': self._drift_event_count_,
                'enable_dynamic_spawning': self.enable_dynamic_spawning,
                # --- runtime counters ---
                'prediction_count': self.prediction_count_,
            }
        }
        
        joblib.dump(save_data, filename)
        logger.info(f"Enhanced model saved to {filename} (architecture_version={self._ARCHITECTURE_VERSION})")
    
    @classmethod
    def load_model(cls, filename: str) -> 'OptimizedTRA':
        """PHASE6 item 29: load a trained model, validating its metadata against the CURRENT
        class's expectations rather than trusting it blindly.

        Never hard-blocks loading on a version/schema mismatch — spec item 38 requires existing
        saved models to keep working — but logs clear, specific warnings so the caller knows
        when a save predates a feature (e.g. a pre-Phase-6 save has no calibration/drift
        metadata at all) rather than silently assuming everything is present.
        """
        save_data = joblib.load(filename)
        
        if isinstance(save_data, dict) and 'model' in save_data:
            model = save_data['model']
            metadata = save_data.get('metadata', {})
            logger.info(f"Enhanced model loaded from {filename}")

            saved_version = metadata.get('architecture_version')
            current_version = getattr(cls, '_ARCHITECTURE_VERSION', None)
            if saved_version is None:
                logger.warning(f"Loaded model has no architecture_version in its metadata "
                                f"(saved before PHASE6 versioning was added). It should still "
                                f"work, but was not saved with the current metadata schema — "
                                f"re-saving with save_model() will upgrade it.")
            elif saved_version != current_version:
                logger.warning(f"Loaded model architecture_version ('{saved_version}') does not "
                                f"match the current code's version ('{current_version}'). The "
                                f"model will still be loaded as-is, but behavior may differ from "
                                f"a model freshly trained with this version of the code if the "
                                f"architecture changed between versions.")
            else:
                logger.debug(f"architecture_version matches current code ({current_version}).")

            saved_n_features = metadata.get('n_features')
            if saved_n_features is not None and getattr(model, 'n_features_in_', None) is not None \
                    and saved_n_features != model.n_features_in_:
                logger.warning(f"Metadata n_features ({saved_n_features}) does not match the "
                                f"loaded model's n_features_in_ ({model.n_features_in_}) — the "
                                f"save file may be corrupted or hand-edited.")

            logger.info(f"Model metadata: {metadata}")
        else:
            # Backward compatibility: a bare pickled model with no metadata wrapper at all
            # (pre-dates even the original metadata dict, let alone PHASE6 versioning).
            model = save_data
            logger.warning(f"Legacy model loaded from {filename} with NO metadata wrapper — "
                            f"this predates architecture versioning entirely. Re-save with "
                            f"save_model() to upgrade to the current metadata schema.")
        
        return model
    
    @staticmethod
    def create_example_dataset(task_type: str = "classification", 
                             n_samples: int = 1000, 
                             n_features: int = 10,
                             random_state: int = 42,
                             noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Create enhanced example dataset for testing with more realistic characteristics."""
        np.random.seed(random_state)
        
        if task_type == "classification":
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(3, n_features // 2),
                n_redundant=max(1, n_features // 4),
                n_classes=3,
                n_clusters_per_class=2,  # More complex clusters
                flip_y=noise_level,  # Add label noise
                random_state=random_state,
                class_sep=0.8  # Moderate class separation
            )
        else:
            from sklearn.datasets import make_regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(3, n_features // 2),
                noise=noise_level * 10,  # Scaled noise for regression
                random_state=random_state,
                bias=10.0  # Add bias term
            )
        
        return X, y

def run_enhanced_example():
    """Run enhanced demonstration of Optimized TRA with all improvements."""
    logger.info("=" * 70)
    logger.info("ENHANCED OPTIMIZED TRACK/RAIL ALGORITHM (TRA) DEMONSTRATION")
    logger.info("=" * 70)
    
    for task_type in ["classification", "regression"]:
        logger.info(f"\n{task_type.upper()} EXAMPLE WITH OPTIMIZATIONS")
        logger.info("-" * 50)
        
        # Create enhanced dataset
        X, y = EnhancedTRA.create_example_dataset(
            task_type=task_type, 
            n_samples=1200,
            n_features=15,
            noise_level=0.15
        )
        logger.info(f"Enhanced dataset created: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data with validation set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Test both routing modes
        for mode in ["hard", "soft"]:
            logger.info(f"\nTesting Routing Mode: {mode.upper()}")
            
            # Create and train enhanced model
            tra = EnhancedTRA(
                task_type=task_type,
                n_tracks=5,  # More tracks for better performance
                random_state=42,
                n_estimators=40,
                max_depth=6,
                feature_selection=True,
                handle_imbalanced=True,
                enable_track_pruning=True,
                pruning_interval=50,
                routing_mode=mode,
                cluster_experts=True
            )
            
            # Train model
            start_time = time.time()
            tra.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = tra.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Evaluate performance
            if task_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                logger.info(f"Test Accuracy: {accuracy:.4f}")
                logger.info(f"Test F1-score: {f1:.4f}")
                
                # Test probability predictions
                try:
                    y_proba = tra.predict_proba(X_test)
                    logger.info(f"Probability predictions shape: {y_proba.shape}")
                except Exception as e:
                    logger.warning(f"Probability prediction failed: {str(e)}")
                    
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                logger.info(f"Test MSE: {mse:.4f}")
                logger.info(f"Test RMSE: {rmse:.4f}")
            
            logger.info(f"Training time: {training_time:.2f}s")
            logger.info(f"Prediction time: {prediction_time:.4f}s ({prediction_time/len(X_test)*1000:.2f}ms per sample)")
        
        # Test enhanced model saving/loading
        model_filename = f"enhanced_tra_{task_type}_model.joblib"
        try:
            tra.save_model(model_filename)
            loaded_tra = EnhancedTRA.load_model(model_filename)
            loaded_pred = loaded_tra.predict(X_test[:10])
            logger.info(f"Enhanced model save/load test successful: {len(loaded_pred)} predictions")
        except Exception as e:
            logger.warning(f"Model save/load failed: {str(e)}")
        
        # Test scoring with timing
        try:
            start_time = time.time()
            score = tra.score(X_test, y_test)
            scoring_time = time.time() - start_time
            logger.info(f"Model score: {score:.4f} (computed in {scoring_time:.4f}s)")
        except Exception as e:
            logger.warning(f"Scoring failed: {str(e)}")

# Alias for backward compatibility
OptimizedTRA = EnhancedTRA


if __name__ == "__main__":
    try:
        run_enhanced_example()
        logger.info("\n" + "=" * 70)
        logger.info("ENHANCED TRA DEMONSTRATION COMPLETED SUCCESSFULLY!")
        logger.info("Key Improvements:")
        logger.info("✓ Data Leakage fixed in MoE Router")
        logger.info("✓ Hard and Soft Routing modes fully implemented")
        logger.info("✓ K-Means Clustering Expert Tracks initialized")
        logger.info("✓ Fully Vectorized Inference Prediction framework")
        logger.info("✓ Automatic track pruning for memory optimization")
        logger.info("✓ Advanced performance monitoring and reporting")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Error during enhanced demonstration: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
