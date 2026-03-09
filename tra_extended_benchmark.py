"""
TRA Algorithm Extended Benchmark Framework
============================================

A comprehensive, reproducible benchmarking framework for evaluating the Track/Rail Algorithm (TRA) 
against strong baselines on diverse datasets with rigorous statistical validation.

Implements:
- 20+ diverse classification and regression datasets
- 8 strong baseline models
- k-fold cross-validation with repeated experiments
- Confidence intervals and statistical significance tests
- Computational complexity analysis
- IEEE-ready output and visualizations

Author: TRA Development Team
Date: March 2026
License: MIT
"""
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

# sklearn imports
from sklearn.model_selection import (
    StratifiedKFold, KFold, RepeatedStratifiedKFold, RepeatedKFold,
    train_test_split, cross_validate
)
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    load_diabetes, fetch_california_housing, fetch_openml
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error,
    mean_absolute_error, r2_score, make_scorer, classification_report
)

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from scipy.stats import ttest_rel, wilcoxon, f_oneway
    from scipy.stats import rankdata, norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable or []
        def __iter__(self):
            return iter(self.iterable)

# TRA import
try:
    from tra_algorithm import OptimizedTRA
    HAS_TRA = True
except ImportError:
    HAS_TRA = False
    logger = logging.getLogger(__name__)
    logger.warning("TRA Algorithm not found. Install: pip install tra-algorithm")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configuration
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ==============================================================================
# SECTION 1: DATA STRUCTURES
# ==============================================================================

@dataclass
class Dataset:
    """Container for dataset metadata and data."""
    name: str
    X: np.ndarray
    y: np.ndarray
    task_type: str  # 'classification' or 'regression'
    n_samples: int = field(init=False)
    n_features: int = field(init=False)
    n_classes: Optional[int] = field(init=False)
    
    def __post_init__(self):
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        if self.task_type == 'classification':
            self.n_classes = len(np.unique(self.y))
        else:
            self.n_classes = None


@dataclass
class ExperimentResult:
    """Container for a single model's performance on a dataset."""
    dataset_name: str
    model_name: str
    task_type: str
    fold: int
    metrics: Dict[str, float]
    train_time: float
    predict_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        """Convert to dictionary for DataFrame storage."""
        result = {
            'dataset': self.dataset_name,
            'model': self.model_name,
            'task_type': self.task_type,
            'fold': self.fold,
            'train_time': self.train_time,
            'predict_time': self.predict_time,
            'timestamp': self.timestamp
        }
        result.update(self.metrics)
        return result


# ==============================================================================
# SECTION 2: DATASET LOADERS
# ==============================================================================

# Classification datasets from OpenML
CLASSIFICATION_OPENML_DATASETS = [
    "adult",
    "bank-marketing",
    "credit-g",
    "phoneme",
    "spambase",
    "magic",
    "electricity",
    "eeg-eye-state",
    "nomao",
    "qsar-biodeg",
    "madelon",
    "connect-4",
    "jungle",
    "jungle"
]

# Regression datasets from OpenML
REGRESSION_OPENML_DATASETS = [
    "energy-efficiency",
    "concrete_compressive_strength",
    "yacht_hydrodynamics",
    "airfoil_self_noise",
    "bike_sharing",
    "house_prices",
    "abalone",
    "superconduct",
    "gas_turbine_co",
    "insurance"
]


def load_openml_dataset(dataset_name: str, task_type: str) -> Optional[Dataset]:
    """Load a dataset from OpenML with robust error handling.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the OpenML dataset
    task_type : str
        'classification' or 'regression'
    
    Returns:
    --------
    Dataset or None
        Loaded dataset or None if loading fails
    """
    try:
        logger.info(f"Loading {dataset_name} ({task_type})...")
        data = fetch_openml(name=dataset_name, as_frame=True, parser='auto')
        
        X = data.data
        y = data.target
        
        # Handle missing values
        if X.isnull().any().any():
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0] if len(X[categorical_cols].mode()) > 0 else 'unknown')
        
        # Handle target missing values
        if isinstance(y, pd.Series) and y.isnull().any():
            y = y.dropna()
            X = X.loc[y.index]
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            encoder = LabelEncoder()
            for col in categorical_cols:
                X[col] = encoder.fit_transform(X[col].astype(str))
        
        # Convert to numpy array
        X_array = X.values
        
        # Handle target encoding
        if task_type == 'classification':
            if isinstance(y, pd.Series):
                y = y.values
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)
            else:
                y = y.astype(int)
        elif task_type == 'regression':
            if isinstance(y, pd.Series):
                y = y.values
            y = pd.to_numeric(y, errors='coerce')
            y = np.nan_to_num(y, nan=np.nanmean(y))  # Fill NaN with mean
        
        # Skip very large datasets
        if X_array.shape[0] > 200000:
            logger.info(f"Skipping {dataset_name}: dataset too large ({X_array.shape[0]} samples)")
            return None
        
        # Ensure numeric array
        X_array = X_array.astype(np.float64)
        
        dataset = Dataset(dataset_name, X_array, y, task_type)
        logger.info(f"Loaded {dataset_name}: {dataset.n_samples} samples, {dataset.n_features} features")
        return dataset
        
    except Exception as e:
        logger.warning(f"Failed to load {dataset_name}: {str(e)[:100]}")
        return None


class DatasetLoader:
    """Loader for diverse dataset collection."""
    
    @staticmethod
    def load_classification_datasets() -> List[Dataset]:
        """Load all classification datasets from sklearn and OpenML."""
        datasets = []

        logger.info("Loading classification datasets...")

        # Built-in sklearn datasets
        try:
            iris = load_iris()
            datasets.append(Dataset("Iris", iris.data, iris.target, "classification"))
        except Exception as e:
            logger.warning(f"Failed to load Iris: {e}")

        try:
            wine = load_wine()
            datasets.append(Dataset("Wine", wine.data, wine.target, "classification"))
        except Exception as e:
            logger.warning(f"Failed to load Wine: {e}")

        try:
            cancer = load_breast_cancer()
            datasets.append(Dataset("Breast Cancer", cancer.data, cancer.target, "classification"))
        except Exception as e:
            logger.warning(f"Failed to load Breast Cancer: {e}")

        try:
            digits = load_digits()
            datasets.append(Dataset("Digits", digits.data, digits.target, "classification"))
        except Exception as e:
            logger.warning(f"Failed to load Digits: {e}")

        # OpenML classification datasets from registry
        for dataset_name in CLASSIFICATION_OPENML_DATASETS:
            dataset = load_openml_dataset(dataset_name, "classification")
            if dataset is not None:
                datasets.append(dataset)

        logger.info(f"Loaded {len(datasets)} classification datasets")
        return datasets
    
    @staticmethod
    def load_regression_datasets() -> List[Dataset]:
        """Load all regression datasets from sklearn and OpenML."""
        datasets = []

        logger.info("Loading regression datasets...")

        # Built-in sklearn datasets
        try:
            diabetes = load_diabetes()
            datasets.append(Dataset("Diabetes", diabetes.data, diabetes.target, "regression"))
        except Exception as e:
            logger.warning(f"Failed to load Diabetes: {e}")

        try:
            housing = fetch_california_housing()
            datasets.append(Dataset("California Housing", housing.data, housing.target, "regression"))
        except Exception as e:
            logger.warning(f"Failed to load California Housing: {e}")

        # OpenML regression datasets from registry
        for dataset_name in REGRESSION_OPENML_DATASETS:
            dataset = load_openml_dataset(dataset_name, "regression")
            if dataset is not None:
                datasets.append(dataset)

        logger.info(f"Loaded {len(datasets)} regression datasets")
        return datasets
    
    @staticmethod
    def load_all_datasets() -> Tuple[List[Dataset], List[Dataset]]:
        """Load all classification and regression datasets."""
        clf_datasets = DatasetLoader.load_classification_datasets()
        reg_datasets = DatasetLoader.load_regression_datasets()
        
        # Log summary statistics
        total_datasets = len(clf_datasets) + len(reg_datasets)
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET LOADING SUMMARY")
        logger.info(f"Classification datasets: {len(clf_datasets)}")
        logger.info(f"Regression datasets: {len(reg_datasets)}")
        logger.info(f"Total datasets: {total_datasets}")
        logger.info(f"{'='*70}\n")
        
        return clf_datasets, reg_datasets


# ==============================================================================
# SECTION 3: MODEL BUILDERS
# ==============================================================================

class ModelBuilder:
    """Factory for creating baseline and TRA models."""
    
    @staticmethod
    def build_classification_models() -> Dict[str, Any]:
        """Build all classification baseline models."""
        models = {}
        
        # Random Forest
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        )
        
        # Gradient Boosting
        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=7, learning_rate=0.1, random_state=RANDOM_STATE
        )
        
        # SVM
        models['SVM'] = SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE)
        
        # KNN
        models['KNN'] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        
        # MLP
        models['MLP'] = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE
        )
        
        # XGBoost
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=7, learning_rate=0.1,
                random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'
            )
        
        # LightGBM
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=7, learning_rate=0.1,
                random_state=RANDOM_STATE, verbose=-1
            )
        
        # CatBoost
        if HAS_CATBOOST:
            models['CatBoost'] = cb.CatBoostClassifier(
                iterations=100, depth=7, learning_rate=0.1,
                random_state=RANDOM_STATE, verbose=False
            )
        
        # TRA Algorithm - OPTIMIZED with all 11 improvements
        if HAS_TRA:
            # TRA_Hard: Efficient single-expert routing
            models['TRA_Hard'] = OptimizedTRA(
                task_type='classification',
                # IMPROVEMENT 3: More diverse expert tracks
                n_tracks=7,
                max_tracks=8,
                # IMPROVEMENT 1: Stronger router
                router_type='catboost' if HAS_CATBOOST else 'xgboost',
                # IMPROVEMENT 5 & 2: Top-K routing with heterogeneous experts
                top_k=1,
                routing_mode='hard',
                # IMPROVEMENT 7: Router meta-features (disagreement signals)
                use_meta_features=True,
                # IMPROVEMENT 10: Track specialization via clustering
                cluster_experts=True,
                # IMPROVEMENT 3: Residual correction track (TRA-Boost)
                enable_correction_track=True,
                # IMPROVEMENT 4: Load balancing
                load_balance_strength=0.01,
                # IMPROVEMENT 9: Dynamic track spawning
                confidence_spawn_threshold=0.15,
                max_dynamic_tracks=2,
                # IMPROVEMENT 6: Expert capacity control
                expert_capacity=None,
                # Feature selection
                feature_selection=True,
                handle_imbalanced=True,
                enable_track_pruning=True,
                # Base estimator properties
                n_estimators=60,
                max_depth=7,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=RANDOM_STATE
            )
            # TRA_Soft: Multi-expert soft routing with temperature scaling
            models['TRA_Soft'] = OptimizedTRA(
                task_type='classification',
                # IMPROVEMENT 3: More diverse expert tracks
                n_tracks=7,
                max_tracks=8,
                # IMPROVEMENT 1: Stronger router
                router_type='catboost' if HAS_CATBOOST else 'xgboost',
                # IMPROVEMENT 5 & 2: Top-K routing with heterogeneous experts
                top_k=2,
                routing_mode='soft',
                # IMPROVEMENT 8: Temperature-scaled soft routing
                routing_temperature=0.8,
                # IMPROVEMENT 7: Router meta-features (disagreement + entropy signals)
                use_meta_features=True,
                # IMPROVEMENT 10: Track specialization via clustering
                cluster_experts=True,
                # IMPROVEMENT 3: Residual correction track (TRA-Boost)
                enable_correction_track=True,
                # IMPROVEMENT 4: Load balancing
                load_balance_strength=0.01,
                # IMPROVEMENT 9: Dynamic track spawning
                confidence_spawn_threshold=0.20,
                max_dynamic_tracks=2,
                # IMPROVEMENT 6: Expert capacity control
                expert_capacity=None,
                # Feature selection
                feature_selection=True,
                handle_imbalanced=True,
                enable_track_pruning=True,
                # Base estimator properties
                n_estimators=60,
                max_depth=7,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=RANDOM_STATE
            )
        
        logger.info(f"Built {len(models)} classification models")
        return models
    
    @staticmethod
    def build_regression_models() -> Dict[str, Any]:
        """Build all regression baseline models."""
        models = {}
        
        # Random Forest
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        )
        
        # Gradient Boosting
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=7, learning_rate=0.1, random_state=RANDOM_STATE
        )
        
        # SVM
        models['SVM'] = SVR(kernel='rbf', C=1.0)
        
        # KNN
        models['KNN'] = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        
        # MLP
        models['MLP'] = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE
        )
        
        # XGBoost
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=7, learning_rate=0.1, random_state=RANDOM_STATE
            )
        
        # LightGBM
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=7, learning_rate=0.1,
                random_state=RANDOM_STATE, verbose=-1
            )
        
        # CatBoost
        if HAS_CATBOOST:
            models['CatBoost'] = cb.CatBoostRegressor(
                iterations=100, depth=7, learning_rate=0.1,
                random_state=RANDOM_STATE, verbose=False
            )
        
        # TRA Algorithm - OPTIMIZED with all 11 improvements
        if HAS_TRA:
            # TRA_Hard: Efficient hard routing for regression
            models['TRA_Hard'] = OptimizedTRA(
                task_type='regression',
                # IMPROVEMENT 3: More diverse expert tracks
                n_tracks=6,
                max_tracks=7,
                # IMPROVEMENT 1: Stronger router
                router_type='lightgbm' if HAS_LIGHTGBM else 'xgboost',
                # IMPROVEMENT 5 & 2: Top-K routing with heterogeneous experts
                top_k=1,
                routing_mode='hard',
                # IMPROVEMENT 7: Router meta-features
                use_meta_features=True,
                # IMPROVEMENT 10: Track specialization via clustering
                cluster_experts=True,
                # IMPROVEMENT 3: Residual correction track (TRA-Boost)
                enable_correction_track=True,
                # IMPROVEMENT 4: Load balancing
                load_balance_strength=0.01,
                # IMPROVEMENT 9: Dynamic track spawning
                confidence_spawn_threshold=0.10,
                max_dynamic_tracks=2,
                # IMPROVEMENT 6: Expert capacity control
                expert_capacity=None,
                # Feature selection
                feature_selection=True,
                enable_track_pruning=True,
                # Base estimator properties (tuned for regression)
                n_estimators=100,
                max_depth=8,
                min_samples_split=8,
                min_samples_leaf=3,
                random_state=RANDOM_STATE
            )
            # TRA_Soft: Multi-expert soft routing for regression
            models['TRA_Soft'] = OptimizedTRA(
                task_type='regression',
                # IMPROVEMENT 3: More diverse expert tracks
                n_tracks=6,
                max_tracks=7,
                # IMPROVEMENT 1: Stronger router
                router_type='lightgbm' if HAS_LIGHTGBM else 'xgboost',
                # IMPROVEMENT 5 & 2: Top-K routing with heterogeneous experts
                top_k=2,
                routing_mode='soft',
                # IMPROVEMENT 8: Temperature-scaled soft routing (smooth for regression)
                routing_temperature=1.0,
                # IMPROVEMENT 7: Router meta-features
                use_meta_features=True,
                # IMPROVEMENT 10: Track specialization via clustering
                cluster_experts=True,
                # IMPROVEMENT 3: Residual correction track (TRA-Boost)
                enable_correction_track=True,
                # IMPROVEMENT 4: Load balancing
                load_balance_strength=0.01,
                # IMPROVEMENT 9: Dynamic track spawning
                confidence_spawn_threshold=0.15,
                max_dynamic_tracks=2,
                # IMPROVEMENT 6: Expert capacity control
                expert_capacity=None,
                # Feature selection
                feature_selection=True,
                enable_track_pruning=True,
                # Base estimator properties (tuned for regression)
                n_estimators=100,
                max_depth=8,
                min_samples_split=8,
                min_samples_leaf=3,
                random_state=RANDOM_STATE
            )
        
        logger.info(f"Built {len(models)} regression models")
        return models


# ==============================================================================
# SECTION 4: EVALUATION METRICS
# ==============================================================================

class MetricsCalculator:
    """Calculate performance metrics for different task types."""
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Failed to compute accuracy: {e}")
            metrics['accuracy'] = 0.0
        
        try:
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except Exception as e:
            logger.warning(f"Failed to compute F1: {e}")
            metrics['f1'] = 0.0
        
        try:
            if y_proba is not None and len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['auc'] = 0.0
        except Exception as e:
            logger.warning(f"Failed to compute AUC: {e}")
            metrics['auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        metrics = {}
        
        try:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        except Exception as e:
            logger.warning(f"Failed to compute RMSE: {e}")
            metrics['rmse'] = float('inf')
        
        try:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Failed to compute MAE: {e}")
            metrics['mae'] = float('inf')
        
        try:
            metrics['r2'] = r2_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Failed to compute R²: {e}")
            metrics['r2'] = 0.0
        
        return metrics


# ==============================================================================
# SECTION 5: CROSS VALIDATION
# ==============================================================================

class CrossValidator:
    """K-fold cross-validation with repeated experiments."""
    
    @staticmethod
    def evaluate_classification(model, X, y, dataset_name, model_name,n_splits=10, n_repeats=1):
        """Evaluate classification model with repeated k-fold CV."""
        results = []
        
        if n_repeats > 1:
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                         random_state=RANDOM_STATE)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        
        fold_idx = 0
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train model
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            
            # Predict
            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            predict_time = (time.perf_counter() - t0) * 1000  # Convert to ms
            
            # Compute metrics
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)
                except:
                    pass
            
            metrics = MetricsCalculator.classification_metrics(y_test, y_pred, y_proba)

            result = ExperimentResult(
                dataset_name=dataset_name,
                model_name=model_name,
                task_type="classification",
                fold=fold_idx,
                metrics=metrics,
                train_time=train_time,
                predict_time=predict_time
            )

            results.append(result)

            fold_idx += 1
        
        return results
    
    @staticmethod
    def evaluate_regression(model, X, y, dataset_name, model_name, n_splits=10, n_repeats=1):
        """Evaluate regression model with repeated k-fold CV."""
        results = []
        
        if n_repeats > 1:
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                              random_state=RANDOM_STATE)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        
        fold_idx = 0
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train model
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            
            # Predict
            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            predict_time = (time.perf_counter() - t0) * 1000  # Convert to ms
            
            # Compute metrics
            metrics = MetricsCalculator.regression_metrics(y_test, y_pred)

            result = ExperimentResult(
                dataset_name=dataset_name,
                model_name=model_name,
                task_type="regression",
                fold=fold_idx,
                metrics=metrics,
                train_time=train_time,
                predict_time=predict_time
            )

            results.append(result)

            fold_idx += 1
        
        return results

# ==============================================================================
# SECTION 6: STATISTICAL ANALYSIS
# ==============================================================================

class StatisticalAnalysis:
    """Statistical tests and confidence intervals."""
    
    @staticmethod
    def compute_confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute mean, std, and 95% confidence interval for scores.
        
        CI = mean ± z * (std / sqrt(n))
        where z = 1.96 for 95% confidence
        """
        mean = np.mean(scores)
        std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        n = len(scores)
        z = norm.ppf((1 + confidence) / 2)  # 1.96 for 95%
        margin = z * (std / np.sqrt(n))
        return mean, std, margin
    
    @staticmethod
    def paired_ttest(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
        """Paired t-test between two models."""
        if not HAS_SCIPY:
            logger.warning("scipy not available for statistical tests")
            return 0.0, 1.0
        
        t_stat, p_value = ttest_rel(scores1, scores2)
        return t_stat, p_value
    
    @staticmethod
    def friedman_test(results_df: pd.DataFrame, metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Friedman test for comparing multiple models.
        
        Returns average ranks and test statistics.
        """
        if not HAS_SCIPY:
            logger.warning("scipy not available for statistical tests")
            return {}
        
        # Pivot table: datasets × models
        pivot = results_df.pivot_table(
            index='dataset', columns='model', values=metric, aggfunc='mean'
        )
        
        # Compute average ranks
        ranks = pivot.rank(axis=1, method='average')
        avg_ranks = ranks.mean()
        
        return {'average_ranks': avg_ranks.to_dict(), 'pivot': pivot}
    
    @staticmethod
    def generate_critical_difference_diagram(avg_ranks: Dict[str, float],
                                            n_datasets: int, alpha: float = 0.05) -> str:
        """Generate critical difference diagram description."""
        # Simplified CD computation (full implementation requires post-hoc test)
        n_models = len(avg_ranks)
        k = n_models
        N = n_datasets
        
        if k > 2:
            cd = 2.576 * np.sqrt(k * (k + 1) / (6 * N))  # 95% confidence
            return f"Critical Difference (CD): {cd:.3f}"
        return "Insufficient models for CD diagram"
def run_statistical_tests(results_df, metric='accuracy'):
        """
        Generate statistical comparison between models:
        - p-value table
        - average ranks
        - Friedman test summary
        """

        logger.info("Running statistical significance tests...")

        if not HAS_SCIPY:
            logger.warning("SciPy not installed. Statistical tests skipped.")
            return None

        # Pivot: dataset × model
        results_df = results_df.dropna(subset=[metric])
        pivot = results_df.pivot_table(
            index='dataset',
            columns='model',
            values=metric,
            aggfunc='mean'
        )

        models = pivot.columns
        datasets = pivot.index

        # ----------------------------
        # 1. Average ranks
        # ----------------------------

        ranks = pivot.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean()

        rank_df = pd.DataFrame({
            "model": avg_ranks.index,
            "average_rank": avg_ranks.values
        }).sort_values("average_rank")

        logger.info("Average ranks computed")

        # ----------------------------
        # 2. Friedman test
        # ----------------------------

        data_for_test = [pivot[m].values for m in models]

        from scipy.stats import friedmanchisquare

        stat, p_value = friedmanchisquare(*data_for_test)

        friedman_result = pd.DataFrame({
            "friedman_statistic": [stat],
            "p_value": [p_value],
            "n_models": [len(models)],
            "n_datasets": [len(datasets)]
        })

        logger.info(f"Friedman test p-value: {p_value}")

        # ----------------------------
        # 3. Pairwise p-value table
        # ----------------------------

        p_matrix = pd.DataFrame(
            np.ones((len(models), len(models))),
            index=models,
            columns=models
        )

        for m1 in models:
            for m2 in models:

                if m1 == m2:
                    continue

                scores1 = pivot[m1]
                scores2 = pivot[m2]

                t_stat, p_val = ttest_rel(scores1, scores2)

                p_matrix.loc[m1, m2] = p_val

        logger.info("Pairwise p-value matrix computed")

        return rank_df, friedman_result, p_matrix


# ==============================================================================
# SECTION 7: EXPERIMENT RUNNER
# ==============================================================================

class BenchmarkRunner:
    """Main benchmark execution engine."""
    
    def __init__(self, n_splits: int = 10, n_repeats: int = 1, output_dir: str = './benchmark_results'):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots directory
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        logger.info(f"Initialized BenchmarkRunner with {n_splits}-fold CV, {n_repeats} repeats")
    
    def run_benchmark(self, max_datasets: Optional[int] = None) -> pd.DataFrame:
        """Run full benchmark on all datasets."""
        logger.info("Starting benchmark run...")
        
        # Load datasets
        clf_datasets, reg_datasets = DatasetLoader.load_all_datasets()
        all_datasets = clf_datasets + reg_datasets
        
        if max_datasets:
            all_datasets = all_datasets[:max_datasets]
        
        logger.info(f"Evaluating on {len(all_datasets)} datasets")
        
        # Build models
        clf_models = ModelBuilder.build_classification_models()
        reg_models = ModelBuilder.build_regression_models()
        
        # Run evaluation
        dataset_pbar = tqdm(all_datasets, desc="Datasets", position=0)
        for dataset in dataset_pbar:
            dataset_pbar.set_description(f"Dataset: {dataset.name}")
            
            # Select appropriate models
            models = clf_models if dataset.task_type == 'classification' else reg_models
            
            model_pbar = tqdm(models.items(), desc="Models", position=1, leave=False)
            for model_name, model in model_pbar:
                model_pbar.set_description(f"Model: {model_name}")
                
                try:
                    # Clone model for fresh fit
                    from sklearn.base import clone
                    model_clone = clone(model)
                    
                    if dataset.task_type == 'classification':
                        fold_results = CrossValidator.evaluate_classification(
                            model_clone, dataset.X, dataset.y,dataset.name,model_name,
                            n_splits=self.n_splits, n_repeats=self.n_repeats
                        )
                    else:
                        fold_results = CrossValidator.evaluate_regression(
                            model_clone, dataset.X, dataset.y, dataset.name, model_name,
                            n_splits=self.n_splits, n_repeats=self.n_repeats
                        )
                    
                    self.results.extend(fold_results)
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {dataset.name}: {e}")
                    continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame([r.to_dict() for r in self.results])
        logger.info(f"Benchmark completed with {len(results_df)} results")
        
        return results_df
    
    def compute_summary_statistics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics per model-dataset pair."""
        logger.info("Computing summary statistics...")
        
        summary_stats = []
        
        for (dataset, model), group in results_df.groupby(['dataset', 'model']):
            task_type = group['task_type'].iloc[0]
            
            # Determine metric columns
            if task_type == 'classification':
                metric_cols = ['accuracy', 'f1', 'auc']
            else:
                metric_cols = ['rmse', 'mae', 'r2']
            
            row = {
                'dataset': dataset,
                'model': model,
                'n_folds': len(group)
            }
            
            for metric in metric_cols:
                if metric in group.columns:
                    scores = group[metric].values
                    mean, std, ci = StatisticalAnalysis.compute_confidence_interval(scores)
                    
                    row[f'{metric}_mean'] = mean
                    row[f'{metric}_std'] = std
                    row[f'{metric}_ci'] = ci
            
            row['train_time_mean'] = group['train_time'].mean()
            row['predict_time_mean'] = group['predict_time'].mean()
            
            summary_stats.append(row)
        
        summary_df = pd.DataFrame(summary_stats)
        logger.info(f"Computed statistics for {len(summary_df)} model-dataset pairs")
        
        return summary_df
    
    def save_results(self, results_df: pd.DataFrame, summary_df: pd.DataFrame):
        """Save results to CSV files."""
        logger.info("Saving results...")
        
        # Full results
        results_file = self.output_dir / 'benchmark_results.csv'
        results_df.to_csv(results_file, index=False)
        logger.info(f"Saved full results to {results_file}")
        
        # Summary statistics
        summary_file = self.output_dir / 'benchmark_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary statistics to {summary_file}")
        
        # Generate text summary
        self._save_text_summary(results_df, summary_df)
    
    def _save_text_summary(self, results_df: pd.DataFrame, summary_df: pd.DataFrame):
        """Save human-readable summary report."""
        report_file = self.output_dir / 'benchmark_summary.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRA ALGORITHM BENCHMARK REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"K-Fold Splits: {self.n_splits}\n")
            f.write(f"Repeats: {self.n_repeats}\n")
            f.write(f"Total Evaluations: {len(results_df)}\n")
            f.write(f"Unique Datasets: {results_df['dataset'].nunique()}\n")
            f.write(f"Unique Models: {results_df['model'].nunique()}\n\n")
            
            # Classification Results
            clf_summary = summary_df[summary_df.index < len(summary_df)]
            if len(clf_summary) > 0:
                f.write("\nCLASSIFICATION RESULTS (Top 10)\n")
                f.write("-" * 80 + "\n")
                top_clf = clf_summary.nlargest(10, 'accuracy_mean') if 'accuracy_mean' in clf_summary.columns else clf_summary.head(10)
                f.write(top_clf.to_string())
                f.write("\n\n")
            
            # Regression Results
            if len(summary_df) > 0 and 'rmse_mean' in summary_df.columns:
                f.write("\nREGRESSION RESULTS (Top 10)\n")
                f.write("-" * 80 + "\n")
                top_reg = summary_df.nsmallest(10, 'rmse_mean')
                f.write(top_reg.to_string())
                f.write("\n\n")
            
            # Model Rankings
            f.write("\nMODEL AVERAGE PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            model_avg = summary_df.groupby('model').agg({
                col: 'mean' for col in summary_df.columns if col not in ['dataset', 'model']
            })
            f.write(model_avg.to_string())
            f.write("\n\n")
            
            # Computational Complexity
            f.write("\nCOMPUTATIONAL COMPLEXITY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            timing_stats = summary_df.groupby('model')[['train_time_mean', 'predict_time_mean']].mean()
            f.write(timing_stats.to_string())
            f.write("\n")
        
        logger.info(f"Saved text summary to {report_file}")
    
    def generate_visualizations(self, summary_df: pd.DataFrame):
        """Generate publication-quality plots."""
        logger.info("Generating visualizations...")
        
        # Plot 1: Model Performance Comparison (Accuracy/RMSE)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Classification accuracy
            if 'accuracy_mean' in summary_df.columns:
                clf_data = summary_df[['model', 'accuracy_mean']].drop_duplicates()
                clf_data = clf_data.sort_values('accuracy_mean', ascending=False)
                axes[0].barh(clf_data['model'], clf_data['accuracy_mean'])
                axes[0].set_xlabel('Mean Accuracy')
                axes[0].set_title('Classification Performance')
                axes[0].set_xlim(0, 1)
            
            # Regression RMSE
            if 'rmse_mean' in summary_df.columns:
                reg_data = summary_df[['model', 'rmse_mean']].drop_duplicates()
                reg_data = reg_data.sort_values('rmse_mean', ascending=True)
                axes[1].barh(reg_data['model'], reg_data['rmse_mean'], color='coral')
                axes[1].set_xlabel('Mean RMSE')
                axes[1].set_title('Regression Performance')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
            logger.info("Saved performance comparison plot")
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate performance plot: {e}")
        
        # Plot 2: Training Time Comparison
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            timing = summary_df.groupby('model')['train_time_mean'].mean().sort_values(ascending=False)
            ax.barh(timing.index, timing.values, color='steelblue')
            ax.set_xlabel('Mean Training Time (seconds)')
            ax.set_title('Training Time Complexity')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'training_time.png', dpi=300, bbox_inches='tight')
            logger.info("Saved training time plot")
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate timing plot: {e}")
        
        # Plot 3: Prediction Time Comparison
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            pred_time = summary_df.groupby('model')['predict_time_mean'].mean().sort_values(ascending=False)
            ax.barh(pred_time.index, pred_time.values, color='seagreen')
            ax.set_xlabel('Mean Prediction Time (ms)')
            ax.set_title('Inference Time Complexity')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'prediction_time.png', dpi=300, bbox_inches='tight')
            logger.info("Saved prediction time plot")
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate prediction time plot: {e}")


# ==============================================================================
# SECTION 8: LATEX TABLE GENERATION
# ==============================================================================

class LatexTableGenerator:
    """Generate publication-ready LaTeX tables."""
    
    @staticmethod
    def generate_results_table(summary_df: pd.DataFrame, metric: str = 'accuracy_mean',
                              top_n: int = 10) -> str:
        """
        Generate LaTeX table for top models.
        
        Example output:
        \\begin{table}[H]
        \\centering
        \\begin{tabular}{lcc}
        ...
        \\end{tabular}
        \\end{table}
        """
        top_results = summary_df.nlargest(top_n, metric)
        
        latex_code = "\\begin{table}[H]\n"
        latex_code += "\\centering\n"
        latex_code += "\\small\n"
        latex_code += "\\begin{tabular}{lccc}\n"
        latex_code += "\\toprule\n"
        latex_code += "Dataset & Model & Accuracy & F1-Score \\\\\n"
        latex_code += "\\midrule\n"
        
        for _, row in top_results.iterrows():
            latex_code += f"{row['dataset']:20s} & {row['model']:15s} & "
            if 'accuracy_mean' in row:
                latex_code += f"{row['accuracy_mean']:.4f} & "
            if 'f1_mean' in row:
                latex_code += f"{row['f1_mean']:.4f}"
            latex_code += " \\\\\n"
        
        latex_code += "\\bottomrule\n"
        latex_code += "\\end{tabular}\n"
        latex_code += "\\caption{Top 10 Classification Results}\n"
        latex_code += "\\label{tab:results}\n"
        latex_code += "\\end{table}\n"
        
        return latex_code
    
    @staticmethod
    def save_latex_table(latex_code: str, filename: str = 'results_table.tex'):
        """Save LaTeX code to file."""
        with open(filename, 'w') as f:
            f.write(latex_code)
        logger.info(f"Saved LaTeX table to {filename}")


# ==============================================================================
# SECTION 9: MAIN EXECUTION
# ==============================================================================

def main():
    """Main benchmark execution function."""
    logger.info("=" * 80)
    logger.info("TRA ALGORITHM EXTENDED BENCHMARK SUITE")
    logger.info("=" * 80)
    
    # Run benchmark
    runner = BenchmarkRunner(n_splits=5, n_repeats=2, output_dir='./benchmark_results')
    results_df = runner.run_benchmark(max_datasets=None)  # Set to small number for quick test
    
    # Compute statistics
    summary_df = runner.compute_summary_statistics(results_df)
    # Run statistical significance analysis
    # Split classification and regression
    classification_results = results_df[results_df["task_type"] == "classification"]
    regression_results = results_df[results_df["task_type"] == "regression"]

    # Classification statistical tests
    rank_df_clf, friedman_df_clf, p_values_clf = run_statistical_tests(
        classification_results, metric="accuracy"
    )

    # Regression statistical tests
    rank_df_reg, friedman_df_reg, p_values_reg = run_statistical_tests(
        regression_results, metric="r2"
    )

    # Save classification statistical results
    if rank_df_clf is not None:
        rank_df_clf.to_csv("benchmark_results/model_ranks_classification.csv", index=False)
        friedman_df_clf.to_csv("benchmark_results/friedman_classification.csv", index=False)
        p_values_clf.to_csv("benchmark_results/p_values_classification.csv")

    # Save regression statistical results
    if rank_df_reg is not None:
        rank_df_reg.to_csv("benchmark_results/model_ranks_regression.csv", index=False)
        friedman_df_reg.to_csv("benchmark_results/friedman_regression.csv", index=False)
        p_values_reg.to_csv("benchmark_results/p_values_regression.csv")

    logger.info("Statistical test results saved.")
    
    # Save results
    runner.save_results(results_df, summary_df)
    
    # Generate visualizations
    runner.generate_visualizations(summary_df)
    
    # Generate LaTeX table
    latex_table = LatexTableGenerator.generate_results_table(summary_df)
    LatexTableGenerator.save_latex_table(latex_table, 'benchmark_results/results_table.tex')
    
    logger.info("=" * 80)
    logger.info("BENCHMARK COMPLETED SUCCESSFULLY")
    logger.info("Results saved to ./benchmark_results/")
    logger.info("=" * 80)
    
    return results_df, summary_df


if __name__ == "__main__":
    results_df, summary_df = main()
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(summary_df.head(20))
    print("\nFor full results, see benchmark_results/")
