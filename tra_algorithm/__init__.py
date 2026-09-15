"""
TRA Algorithm Package - Enhanced Track/Rail Algorithm
======================================================

A sophisticated Mixture-of-Experts (MoE) ensemble machine learning architecture 
that combines Switch Transformer-inspired routing with signal-guided expert gating.

The Extended TRA algorithm uses multiple heterogeneous expert "tracks" with 
intelligent "signal-guided routing" to dynamically select the best experts, 
providing improved performance for both classification and regression tasks.

Example Usage:
-------------
    from tra_algorithm import OptimizedTRA
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize and train TRA with MoE routing
    tra = OptimizedTRA(
        task_type="classification",
        n_tracks=5,
        router_type="xgboost",
        routing_mode="soft"
    )
    tra.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tra.predict(X_test)
    y_proba = tra.predict_proba(X_test)
    
    # Evaluate performance
    score = tra.score(X_test, y_test)
    print(f"Accuracy: {score:.4f}")

Core Classes:
-------------
    OptimizedTRA: Main MoE algorithm (alias for EnhancedTRA)
        - Mixture-of-Experts architecture with 5-8+ expert tracks
        - Signal-guided routing with 5 structural signals
        - Soft & hard routing modes with temperature scaling
        - Out-of-core learning via partial_fit()
        - Automatic track pruning and dynamic expert spawning
    
    EnhancedTRA: Enhanced version with all 11 improvements integrated
        - All features identical to OptimizedTRA
    
    Track: Individual expert model with performance monitoring
        - Prediction time tracking
        - Usage counters
        - Pruning logic
    
    SignalExtractor: Structural signal extraction layer
        - Expert disagreement (prediction std)
        - Prediction entropy
        - Feature density (k-NN based)
        - Cluster distance (KMeans)
        - Outlier score (IsolationForest)
    
Utilities:
----------
    create_example_dataset: Generate synthetic datasets for testing
    evaluate_model_performance: Compute classification/regression metrics
    plot_learning_curves: Visualize learning curves with sklearn framework
    compare_with_baselines: Compare TRA against baseline models
    
Examples:
---------
    basic_classification_example, basic_regression_example, model_comparison_example
"""

from .version import __version__
from .core import (
    OptimizedTRA,
    Track,
    EnhancedTRA
)

from .utils import (
    create_example_dataset,
    evaluate_model_performance,
    plot_learning_curves,
    compare_with_baselines
)
from .examples import (
    basic_classification_example,
    basic_regression_example,
    model_comparison_example
)

__all__ = [
    # Main classes
    'OptimizedTRA',
    'EnhancedTRA',  # Enhanced version with MoE improvements
    'Track',
    
    # Utilities
    'create_example_dataset',
    'evaluate_model_performance',
    'plot_learning_curves',
    'compare_with_baselines',
    
    # Examples
    'basic_classification_example',
    'basic_regression_example', 
    'model_comparison_example',
    
    # Version
    '__version__',
]

# Package metadata
__author__ = "TRA Algorithm Team"
__email__ = "contact@tra-algorithm.com"
__license__ = "MIT"
__description__ = "Track/Rail Algorithm (TRA) - A novel machine learning algorithm for dynamic model selection"