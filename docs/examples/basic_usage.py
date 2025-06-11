#!/usr/bin/env python3
"""
Basic Usage Example for TRA Algorithm Package

This example demonstrates how to use the Track/Rail Algorithm (TRA) 
for both classification and regression tasks.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Import TRA algorithm
from tra_algorithm import OptimizedTRA


def classification_example():
    """Demonstrate TRA usage for classification."""
    print("=" * 60)
    print("TRA CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Create sample dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train TRA model
    tra = OptimizedTRA(
        task_type="classification",
        n_tracks=3,
        signal_threshold=0.1,
        random_state=42
    )
    
    print("\nTraining TRA model...")
    tra.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tra.predict(X_test)
    y_proba = tra.predict_proba(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Probability predictions shape: {y_proba.shape}")
    
    # Display model statistics
    stats = tra.get_track_statistics()
    print(f"\nModel Statistics:")
    print(f"Number of tracks: {stats['n_tracks']}")
    print(f"Number of signals: {stats['n_signals']}")
    
    return tra


def regression_example():
    """Demonstrate TRA usage for regression."""
    print("\n" + "=" * 60)
    print("TRA REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Create sample dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=0.1,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train TRA model
    tra = OptimizedTRA(
        task_type="regression",
        n_tracks=4,
        signal_threshold=0.15,
        random_state=42
    )
    
    print("\nTraining TRA model...")
    tra.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tra.predict(X_test)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nResults:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Display model statistics
    stats = tra.get_track_statistics()
    print(f"\nModel Statistics:")
    print(f"Number of tracks: {stats['n_tracks']}")
    print(f"Number of signals: {stats['n_signals']}")
    
    return tra


def advanced_usage_example():
    """Demonstrate advanced TRA features."""
    print("\n" + "=" * 60)
    print("TRA ADVANCED FEATURES EXAMPLE")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=800,
        n_features=15,
        n_informative=10,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create advanced TRA model with all features enabled
    tra = OptimizedTRA(
        task_type="classification",
        n_tracks=5,
        signal_threshold=0.12,
        random_state=42,
        feature_selection=True,
        handle_imbalanced=True,
        parallel_signals=True,
        enable_track_pruning=True,
        max_workers=4
    )
    
    print("Training advanced TRA model with all features...")
    tra.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tra.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate performance report
    print("\nPerformance Report:")
    print(tra.get_performance_report())
    
    # Save model
    model_filename = "advanced_tra_model.joblib"
    tra.save_model(model_filename)
    print(f"\nModel saved to {model_filename}")
    
    # Load model and test
    loaded_tra = OptimizedTRA.load_model(model_filename)
    loaded_pred = loaded_tra.predict(X_test[:5])
    print(f"Loaded model prediction test: {len(loaded_pred)} predictions made")
    
    return tra


def main():
    """Run all examples."""
    print("TRA ALGORITHM - BASIC USAGE EXAMPLES")
    print("====================================")
    
    # Run classification example
    clf_tra = classification_example()
    
    # Run regression example
    reg_tra = regression_example()
    
    # Run advanced features example
    adv_tra = advanced_usage_example()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nQuick Start Guide:")
    print("1. Import: from tra_algorithm import OptimizedTRA")
    print("2. Create: tra = OptimizedTRA(task_type='classification')")
    print("3. Train: tra.fit(X_train, y_train)")
    print("4. Predict: y_pred = tra.predict(X_test)")
    print("5. Evaluate: tra.get_performance_report()")


if __name__ == "__main__":
    main()