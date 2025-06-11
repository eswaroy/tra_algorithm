"""
TRA Algorithm Examples
=====================

This module contains comprehensive examples demonstrating how to use the 
Track/Rail Algorithm (TRA) for various machine learning tasks.

Examples include:
- Basic classification and regression
- Advanced configuration options
- Performance optimization
- Custom datasets
- Model evaluation and visualization
- Real-world use cases

Author: TRA Development Team
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import (
    make_classification, make_regression, load_iris, load_wine,
    load_diabetes, fetch_california_housing
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, mean_squared_error,
    r2_score, accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
import time
import logging

# Import TRA algorithm (assuming it's in the same package)
try:
    from core import OptimizedTRA
except ImportError:
    from tra_algorithm.core import OptimizedTRA

# Configure logging for examples
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def basic_classification_example():
    """
    Basic classification example using TRA algorithm.
    
    This example demonstrates:
    - Creating a synthetic classification dataset
    - Training TRA classifier
    - Making predictions and evaluating performance
    """
    print("=" * 60)
    print("BASIC CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create and train TRA classifier
    print("\nTraining TRA Classifier...")
    tra_clf = OptimizedTRA(
        task_type="classification",
        n_tracks=4,
        random_state=42,
        n_estimators=50
    )
    
    start_time = time.time()
    tra_clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = tra_clf.predict(X_test)
    y_proba = tra_clf.predict_proba(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nResults:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Display track statistics
    stats = tra_clf.get_track_statistics()
    print(f"\nTRA Statistics:")
    print(f"Number of tracks: {stats['n_tracks']}")
    print(f"Number of signals: {stats['n_signals']}")
    
    return tra_clf, X_test, y_test


def basic_regression_example():
    """
    Basic regression example using TRA algorithm.
    
    This example demonstrates:
    - Creating a synthetic regression dataset
    - Training TRA regressor
    - Making predictions and evaluating performance
    """
    print("\n" + "=" * 60)
    print("BASIC REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Create synthetic dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=0.1,
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train TRA regressor
    print("\nTraining TRA Regressor...")
    tra_reg = OptimizedTRA(
        task_type="regression",
        n_tracks=4,
        random_state=42,
        n_estimators=50
    )
    
    start_time = time.time()
    tra_reg.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = tra_reg.predict(X_test)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Display track statistics
    stats = tra_reg.get_track_statistics()
    print(f"\nTRA Statistics:")
    print(f"Number of tracks: {stats['n_tracks']}")
    print(f"Number of signals: {stats['n_signals']}")
    
    return tra_reg, X_test, y_test


def real_world_classification_example():
    """
    Real-world classification example using the Wine dataset.
    
    This example demonstrates:
    - Loading and preprocessing real data
    - Comparing TRA with traditional methods
    - Advanced configuration options
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD CLASSIFICATION EXAMPLE (Wine Dataset)")
    print("=" * 60)
    
    # Load wine dataset
    wine_data = load_wine()
    X, y = wine_data.data, wine_data.target
    
    print(f"Dataset: {wine_data.DESCR.split('.')[0]}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {wine_data.target_names}")
    print(f"Features: {len(wine_data.feature_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create TRA with advanced configuration
    print("\nTraining Advanced TRA Classifier...")
    tra_clf = OptimizedTRA(
        task_type="classification",
        n_tracks=5,
        signal_threshold=0.1,
        random_state=42,
        n_estimators=100,
        max_depth=8,
        feature_selection=True,
        handle_imbalanced=True,
        parallel_signals=True,
        enable_track_pruning=True
    )
    
    # Train TRA
    start_time = time.time()
    tra_clf.fit(X_train, y_train)
    tra_training_time = time.time() - start_time
    
    # Compare with Random Forest
    print("Training Random Forest for comparison...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    
    start_time = time.time()
    rf_clf.fit(X_train, y_train)
    rf_training_time = time.time() - start_time
    
    # Make predictions
    tra_pred = tra_clf.predict(X_test)
    rf_pred = rf_clf.predict(X_test)
    
    # Evaluate both models
    tra_accuracy = accuracy_score(y_test, tra_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"\nComparison Results:")
    print(f"TRA Accuracy: {tra_accuracy:.4f} (Training time: {tra_training_time:.2f}s)")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f} (Training time: {rf_training_time:.2f}s)")
    
    # Show TRA performance report
    print(f"\nTRA Performance Report:")
    print(tra_clf.get_performance_report())
    
    return tra_clf, rf_clf, X_test, y_test


def real_world_regression_example():
    """
    Real-world regression example using the California Housing dataset.
    
    This example demonstrates:
    - Handling larger real-world datasets
    - Parameter optimization
    - Model evaluation and comparison
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD REGRESSION EXAMPLE (California Housing)")
    print("=" * 60)
    
    # Load California housing dataset
    try:
        housing_data = fetch_california_housing()
        X, y = housing_data.data, housing_data.target
        # Reduce dataset size for speed
        X, y = X[:2000], y[:2000]
    except Exception as e:
        print(f"Could not load California Housing dataset: {e}")
        print("Using synthetic dataset instead...")
        X, y = make_regression(
            n_samples=1000,
            n_features=8,
            n_informative=6,
            noise=0.1,
            random_state=42
        )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target statistics: mean={y.mean():.2f}, std={y.std():.2f}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    # Create and train TRA regressor (faster config)
    print("\nTraining TRA Regressor with optimization...")
    tra_reg = OptimizedTRA(
        task_type="regression",
        n_tracks=2,
        signal_threshold=0.15,
        random_state=42,
        n_estimators=10,
        feature_selection=True,
        parallel_signals=True,
        enable_track_pruning=True
    )
    
    start_time = time.time()
    tra_reg.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Optimize parameters using validation set
    print("Optimizing parameters...")
    optimization_results = tra_reg.optimize_parameters(X_val, y_val)
    
    # Compare with Random Forest
    print("Training Random Forest for comparison...")
    rf_reg = RandomForestRegressor(
        n_estimators=10,
        random_state=42
    )
    
    start_time = time.time()
    rf_reg.fit(X_train, y_train)
    rf_training_time = time.time() - start_time
    
    # Make predictions
    tra_pred = tra_reg.predict(X_test)
    rf_pred = rf_reg.predict(X_test)
    
    # Evaluate both models
    tra_mse = mean_squared_error(y_test, tra_pred)
    tra_r2 = r2_score(y_test, tra_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"\nComparison Results:")
    print(f"TRA - MSE: {tra_mse:.4f}, R²: {tra_r2:.4f} (Training: {training_time:.2f}s)")
    print(f"Random Forest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f} (Training: {rf_training_time:.2f}s)")
    
    print(f"\nParameter Optimization Results:")
    print(f"Original threshold: {optimization_results['original_threshold']}")
    print(f"Optimized threshold: {optimization_results['optimized_threshold']}")
    print(f"Performance improvement: {optimization_results['improvement']:.4f}")
    
    return tra_reg, rf_reg, X_test, y_test


def parameter_tuning_example():
    """
    Example demonstrating parameter tuning for TRA algorithm.
    
    This example shows:
    - Grid search for optimal parameters
    - Cross-validation
    - Performance comparison across different configurations
    """
    print("\n" + "=" * 60)
    print("PARAMETER TUNING EXAMPLE")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=800,
        n_features=12,
        n_informative=10,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset shape: {X.shape}")
    print("Testing different parameter combinations...")
    
    # Define parameter combinations to test
    param_combinations = [
        {'n_tracks': 3, 'signal_threshold': 0.1, 'n_estimators': 50},
        {'n_tracks': 4, 'signal_threshold': 0.1, 'n_estimators': 50},
        {'n_tracks': 5, 'signal_threshold': 0.1, 'n_estimators': 50},
        {'n_tracks': 4, 'signal_threshold': 0.05, 'n_estimators': 50},
        {'n_tracks': 4, 'signal_threshold': 0.15, 'n_estimators': 50},
        {'n_tracks': 4, 'signal_threshold': 0.1, 'n_estimators': 30},
        {'n_tracks': 4, 'signal_threshold': 0.1, 'n_estimators': 70},
    ]
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nTesting configuration {i+1}: {params}")
        
        # Create TRA with current parameters
        tra_clf = OptimizedTRA(
            task_type="classification",
            random_state=42,
            **params
        )
        
        # Train and evaluate
        start_time = time.time()
        tra_clf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = tra_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'params': params,
            'accuracy': accuracy,
            'training_time': training_time
        })
        
        print(f"Accuracy: {accuracy:.4f}, Training time: {training_time:.2f}s")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print(f"\nBest Configuration:")
    print(f"Parameters: {best_result['params']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Training time: {best_result['training_time']:.2f}s")
    
    return results


def model_comparison_example():
    """
    Comprehensive model comparison example.
    
    This example compares TRA with other popular algorithms:
    - Random Forest
    - Decision Tree
    - Gradient Boosting (if available)
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Create challenging dataset
    X, y = make_classification(
        n_samples=1500,
        n_features=15,
        n_informative=12,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {len(np.unique(y))}")
    
    # Initialize models
    models = {
        'TRA': OptimizedTRA(
            task_type="classification",
            n_tracks=5,
            random_state=42,
            n_estimators=80,
            feature_selection=True,
            parallel_signals=True
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=80,
            random_state=42
        )
    }
    
    # Add other models if available
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=80,
            random_state=42
        )
    except:
        pass
    
    try:
        from sklearn.tree import DecisionTreeClassifier
        models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        )
    except:
        pass
    
    # Train and evaluate all models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'prediction_time': prediction_time
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Prediction time: {prediction_time:.4f}s")
    
    # Summary comparison
    print(f"\n{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Train Time':<12} {'Pred Time':<12}")
    print("-" * 70)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} "
              f"{metrics['training_time']:<12.2f} {metrics['prediction_time']:<12.4f}")
    
    return results


def visualization_example():
    """
    Example demonstrating TRA visualization capabilities.
    
    This example shows:
    - Track structure visualization
    - Performance metrics plotting
    - Model comparison charts
    """
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=800,
        n_features=8,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train TRA model
    print("Training TRA model for visualization...")
    tra_clf = OptimizedTRA(
        task_type="classification",
        n_tracks=4,
        random_state=42
    )
    tra_clf.fit(X_train, y_train)
    
    # Make some predictions to generate activity
    _ = tra_clf.predict(X_test)
    
    try:
        # Visualize TRA structure
        print("Creating TRA structure visualization...")
        tra_clf.visualize("tra_structure_example.png")
        print("Visualization saved as 'tra_structure_example.png'")
        
        # Get and display statistics
        stats = tra_clf.get_track_statistics()
        
        # Create performance comparison plot
        plt.figure(figsize=(12, 8))
        
        # Track usage plot
        plt.subplot(2, 2, 1)
        track_names = list(stats['track_details'].keys())
        usage_counts = [stats['track_details'][name]['usage_count'] for name in track_names]
        
        plt.bar(track_names, usage_counts)
        plt.title('Track Usage Distribution')
        plt.xlabel('Track')
        plt.ylabel('Usage Count')
        plt.xticks(rotation=45)
        
        # Performance scores plot
        plt.subplot(2, 2, 2)
        perf_scores = [stats['track_details'][name]['performance_score'] for name in track_names]
        
        plt.bar(track_names, perf_scores)
        plt.title('Track Performance Scores')
        plt.xlabel('Track')
        plt.ylabel('Performance Score')
        plt.xticks(rotation=45)
        
        # Signal confidence plot
        plt.subplot(2, 2, 3)
        signal_conf = [stats['track_details'][name]['avg_signal_confidence'] for name in track_names]
        
        plt.bar(track_names, signal_conf)
        plt.title('Average Signal Confidence')
        plt.xlabel('Track')
        plt.ylabel('Confidence')
        plt.xticks(rotation=45)
        
        # Prediction time plot
        plt.subplot(2, 2, 4)
        pred_times = [stats['track_details'][name]['avg_prediction_time'] * 1000 for name in track_names]
        
        plt.bar(track_names, pred_times)
        plt.title('Average Prediction Time')
        plt.xlabel('Track')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('tra_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Performance metrics plot saved as 'tra_performance_metrics.png'")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("This might be due to missing dependencies (matplotlib, networkx)")
    
    return tra_clf


def custom_dataset_example():
    """
    Example showing how to use TRA with custom datasets.
    
    This example demonstrates:
    - Loading custom data from CSV
    - Data preprocessing
    - Handling categorical variables
    - Model training and evaluation
    """
    print("\n" + "=" * 60)
    print("CUSTOM DATASET EXAMPLE")
    print("=" * 60)
    
    # Create a sample custom dataset (simulating loaded CSV data)
    print("Creating sample custom dataset...")
    
    # Simulate a customer churn dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Numerical features
    age = np.random.normal(35, 12, n_samples)
    income = np.random.lognormal(10, 0.5, n_samples)
    tenure = np.random.exponential(2, n_samples)
    
    # Categorical features (encoded as numbers for simplicity)
    region = np.random.choice([0, 1, 2, 3], n_samples)  # 4 regions
    plan_type = np.random.choice([0, 1, 2], n_samples)  # 3 plan types
    
    # Create target variable (churn) based on features
    churn_prob = (
        -0.02 * age +
        -0.00001 * income +
        -0.1 * tenure +
        0.1 * region +
        0.05 * plan_type +
        np.random.normal(0, 0.1, n_samples)
    )
    churn = (churn_prob > np.median(churn_prob)).astype(int)
    
    # Combine features
    X = np.column_stack([age, income, tenure, region, plan_type])
    y = churn
    
    # Create feature names
    feature_names = ['age', 'income', 'tenure', 'region', 'plan_type']
    
    print(f"Custom dataset created:")
    print(f"Shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Churn rate: {y.mean():.3f}")
    
    # Data preprocessing example
    print("\nPreprocessing data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale numerical features (first 3 columns)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[:, :3] = scaler.fit_transform(X_train[:, :3])
    X_test_scaled[:, :3] = scaler.transform(X_test[:, :3])
    
    print("Numerical features scaled")
    
    # Train TRA model
    print("\nTraining TRA on custom dataset...")
    tra_clf = OptimizedTRA(
        task_type="classification",
        n_tracks=4,
        signal_threshold=0.12,
        random_state=42,
        handle_imbalanced=True,  # Important for imbalanced datasets
        feature_selection=True
    )
    
    tra_clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = tra_clf.predict(X_test_scaled)
    y_proba = tra_clf.predict_proba(X_test_scaled)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nResults on custom dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Show TRA-specific insights
    print(f"\nTRA Model Insights:")
    stats = tra_clf.get_track_statistics()
    print(f"Active tracks: {stats['n_tracks']}")
    print(f"Total signals: {stats['n_signals']}")
    
    for track_name, details in stats['track_details'].items():
        if details['usage_count'] > 0:
            print(f"{track_name}: {details['usage_percentage']:.1f}% usage, "
                  f"performance: {details['performance_score']:.3f}")
    
    return tra_clf, X_test_scaled, y_test


def save_load_example():
    """
    Example demonstrating model saving and loading.
    
    This example shows:
    - Training a TRA model
    - Saving the model to disk
    - Loading the model back
    - Verifying consistency
    """
    print("\n" + "=" * 60)
    print("MODEL SAVE/LOAD EXAMPLE")
    print("=" * 60)
    
    # Create and train model
    X, y = make_classification(n_samples=500, n_features=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Training original TRA model...")
    original_tra = OptimizedTRA(
        task_type="classification",
        n_tracks=3,
        random_state=42
    )
    original_tra.fit(X_train, y_train)
    
    # Make predictions with original model
    original_pred = original_tra.predict(X_test)
    original_accuracy = accuracy_score(y_test, original_pred)
    
    print(f"Original model accuracy: {original_accuracy:.4f}")
    
    # Save model
    model_filename = "tra_model_example.joblib"
    print(f"\nSaving model to {model_filename}...")
    original_tra.save_model(model_filename)
    
    # Load model
    print(f"Loading model from {model_filename}...")
    loaded_tra = OptimizedTRA.load_model(model_filename)
    
    # Make predictions with loaded model
    loaded_pred = loaded_tra.predict(X_test)
    loaded_accuracy = accuracy_score(y_test, loaded_pred)
    
    print(f"Loaded model accuracy: {loaded_accuracy:.4f}")
    
    # Verify consistency
    predictions_match = np.array_equal(original_pred, loaded_pred)
    print(f"Predictions match: {predictions_match}")
    
    if predictions_match:
        print("✓ Model save/load successful!")
    else:
        print("✗ Model save/load failed - predictions don't match")
    
    # Clean up
    import os
    try:
        os.remove(model_filename)
        print(f"Cleaned up {model_filename}")
    except:
        pass
    
    return original_tra, loaded_tra


def run_all_examples():
    """
    Run all examples in sequence.
    
    This function executes all available examples to demonstrate
    the full capabilities of the TRA algorithm.
    """
    print("*" * 80)
    print("RUNNING ALL TRA ALGORITHM EXAMPLES")
    print("*" * 80)
    
    examples = [
        ("Basic Classification Example", basic_classification_example),
        ("Basic Regression Example", basic_regression_example),
        ("Real-World Classification Example", real_world_classification_example),
        ("Real-World Regression Example", real_world_regression_example),
        ("Parameter Tuning Example", parameter_tuning_example),
        ("Model Comparison Example", model_comparison_example),
        ("Visualization Example", visualization_example),
        ("Custom Dataset Example", custom_dataset_example),
        ("Model Save/Load Example", save_load_example)  
    ]
    for name, example_func in examples:
        print(f"\nRunning: {name}")
        try:
            example_func()
            print(f"{name} completed successfully!")
        except Exception as e:
            print(f"Error in {name}: {e}")  
    print("\n" + "*" * 80)
    print("ALL EXAMPLES COMPLETED!")
    print("*" * 80)
    return examples

def main():
    """
    Main function to run all examples.
    
    This serves as the entry point for executing the TRA algorithm examples.
    """
    print("TRA Algorithm Examples")
    print("======================")
    run_all_examples()
    print("\nQuick Start Guide:")
    print("1. Import: from tra_algorithm import OptimizedTRA")
    print("2. Create: tra = OptimizedTRA(task_type='classification')")
    print("3. Train: tra.fit(X_train, y_train)")
    print("4. Predict: y_pred = tra.predict(X_test)")
    print("5. Evaluate: tra.get_performance_report()")
if __name__ == "__main__":
    main()