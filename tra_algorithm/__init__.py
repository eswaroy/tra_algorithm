"""
TRA Algorithm Package
====================

Track/Rail Algorithm (TRA) - A novel machine learning algorithm for dynamic model selection.

The TRA algorithm uses multiple "tracks" (models) and "signals" (switching conditions) to
dynamically route data through the most appropriate model, providing improved performance
for both classification and regression tasks.

Example Usage:
-------------
    from tra_algorithm import OptimizedTRA
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize and train TRA
    tra = OptimizedTRA(task_type="classification", n_tracks=3)
    tra.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tra.predict(X_test)
    
    # Get performance report
    print(tra.get_performance_report())

Classes:
--------
    OptimizedTRA: Main TRA algorithm implementation
    Track: Individual model track
    Signal: Track switching signal
    Record: Data record for tracking
    EnhancedSignalCondition: Advanced signal logic
    
Utilities:
----------
    create_example_dataset: Generate sample datasets for testing
    evaluate_model_performance: Model evaluation utilities
    plot_learning_curves: Learning curve visualization
    compare_with_baselines: Baseline model comparison
    
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

try:
    from .core_enhanced import EnhancedTRA as EnhancedTRANew
except ImportError:
    EnhancedTRANew = EnhancedTRA

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