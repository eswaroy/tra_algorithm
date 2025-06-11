"""
Utility functions for TRA Algorithm package.

This module provides helper functions for dataset creation, model evaluation,
visualization, and comparison with baseline algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from typing import Tuple, Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def create_example_dataset(task_type: str = "classification", 
                          n_samples: int = 1000, 
                          n_features: int = 10,
                          random_state: int = 42,
                          noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create example dataset for testing TRA algorithm.
    
    Parameters:
    -----------
    task_type : str, default="classification"
        Type of task: "classification" or "regression"
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=10
        Number of features to generate
    random_state : int, default=42
        Random state for reproducibility
    noise_level : float, default=0.1
        Amount of noise to add to the data
        
    Returns:
    --------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    """
    np.random.seed(random_state)
    
    if task_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(3, n_features // 2),
            n_redundant=max(1, n_features // 4),
            n_classes=3,
            n_clusters_per_class=2,
            flip_y=noise_level,
            random_state=random_state,
            class_sep=0.8
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(3, n_features // 2),
            noise=noise_level * 10,
            random_state=random_state,
            bias=10.0
        )
    
    return X, y


def evaluate_model_performance(model, X_test: np.ndarray, y_test: np.ndarray, 
                             task_type: str = "classification") -> Dict[str, float]:
    """
    Evaluate model performance with appropriate metrics.
    
    Parameters:
    -----------
    model : estimator
        Trained model to evaluate
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
    task_type : str, default="classification"
        Type of task: "classification" or "regression"
        
    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary of performance metrics
    """
    y_pred = model.predict(X_test)
    
    if task_type == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
        }
        
        # Add probability-based metrics if available
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                # Calculate log loss
                from sklearn.metrics import log_loss
                metrics['log_loss'] = log_loss(y_test, y_proba)
            except:
                pass
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
        }
    
    return metrics


def plot_learning_curves(model, X: np.ndarray, y: np.ndarray, 
                        task_type: str = "classification",
                        cv: int = 5, figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> None:
    """
    Plot learning curves for a model.
    
    Parameters:
    -----------
    model : estimator
        Model to evaluate
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    task_type : str, default="classification"
        Type of task: "classification" or "regression"
    cv : int, default=5
        Number of cross-validation folds
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    # Determine scoring metric
    scoring = 'accuracy' if task_type == "classification" else 'neg_mean_squared_error'
    
    # Calculate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(f'Learning Curves - {model.__class__.__name__}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def compare_with_baselines(tra_model, X: np.ndarray, y: np.ndarray,
                          task_type: str = "classification",
                          test_size: float = 0.2,
                          random_state: int = 42) -> pd.DataFrame:
    """
    Compare TRA model with baseline algorithms.
    
    Parameters:
    -----------
    tra_model : OptimizedTRA
        Trained TRA model
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    task_type : str, default="classification"
        Type of task: "classification" or "regression"
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    results : pd.DataFrame
        Comparison results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Define baseline models
    if task_type == "classification":
        baselines = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'SVM': SVC(random_state=random_state, probability=True)
        }
    else:
        baselines = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Linear Regression': LinearRegression(),
            'SVR': SVR()
        }
    
    results = []
    
    # Evaluate TRA model
    tra_metrics = evaluate_model_performance(tra_model, X_test, y_test, task_type)
    tra_result = {'Model': 'TRA (Optimized)', **tra_metrics}
    results.append(tra_result)
    
    # Evaluate baseline models
    for name, model in baselines.items():
        try:
            model.fit(X_train, y_train)
            metrics = evaluate_model_performance(model, X_test, y_test, task_type)
            result = {'Model': name, **metrics}
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to evaluate {name}: {str(e)}")
    
    return pd.DataFrame(results)


def visualize_performance_comparison(results_df: pd.DataFrame, 
                                   task_type: str = "classification",
                                   figsize: Tuple[int, int] = (12, 6),
                                   save_path: Optional[str] = None) -> None:
    """
    Visualize performance comparison between models.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from compare_with_baselines
    task_type : str, default="classification"
        Type of task: "classification" or "regression"
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    if task_type == "classification":
        metric_cols = ['accuracy', 'f1_score']
        metric_names = ['Accuracy', 'F1 Score']
    else:
        metric_cols = ['r2_score', 'rmse']
        metric_names = ['R² Score', 'RMSE']
    
    fig, axes = plt.subplots(1, len(metric_cols), figsize=figsize)
    if len(metric_cols) == 1:
        axes = [axes]
    
    for i, (col, name) in enumerate(zip(metric_cols, metric_names)):
        if col in results_df.columns:
            bars = axes[i].bar(results_df['Model'], results_df[col])
            axes[i].set_title(f'{name} Comparison')
            axes[i].set_ylabel(name)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Highlight TRA model
            for j, bar in enumerate(bars):
                if 'TRA' in results_df.iloc[j]['Model']:
                    bar.set_color('red')
                    bar.set_alpha(0.8)
                else:
                    bar.set_color('blue')
                    bar.set_alpha(0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def generate_performance_report(tra_model, results_df: pd.DataFrame) -> str:
    """
    Generate a comprehensive performance report.
    
    Parameters:
    -----------
    tra_model : OptimizedTRA
        Trained TRA model
    results_df : pd.DataFrame
        Comparison results
        
    Returns:
    --------
    report : str
        Formatted performance report
    """
    report = []
    report.append("=" * 60)
    report.append("TRA ALGORITHM PERFORMANCE REPORT")
    report.append("=" * 60)
    
    # Model configuration
    report.append(f"Task Type: {tra_model.task_type}")
    report.append(f"Number of Tracks: {len(tra_model.tracks)}")
    report.append(f"Feature Selection: {'Enabled' if tra_model.feature_selection else 'Disabled'}")
    report.append(f"Parallel Processing: {'Enabled' if tra_model.parallel_signals else 'Disabled'}")
    report.append("")
    
    # Performance comparison
    report.append("PERFORMANCE COMPARISON:")
    report.append("-" * 30)
    
    for _, row in results_df.iterrows():
        report.append(f"Model: {row['Model']}")
        for col in results_df.columns:
            if col != 'Model' and pd.notna(row[col]):
                report.append(f"  {col}: {row[col]:.4f}")
        report.append("")
    
    # TRA-specific statistics
    if hasattr(tra_model, 'get_track_statistics'):
        stats = tra_model.get_track_statistics()
        report.append("TRA TRACK STATISTICS:")
        report.append("-" * 30)
        report.append(f"Total Predictions: {stats.get('total_predictions', 0)}")
        report.append(f"Number of Signals: {stats.get('n_signals', 0)}")
        
        if 'track_details' in stats:
            for track_name, details in stats['track_details'].items():
                usage_pct = details.get('usage_percentage', 0)
                report.append(f"Track {track_name}: {usage_pct:.1f}% usage")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def save_model_artifacts(tra_model, results_df: pd.DataFrame, 
                        output_dir: str = "tra_results") -> None:
    """
    Save model artifacts including model, results, and reports.
    
    Parameters:
    -----------
    tra_model : OptimizedTRA
        Trained TRA model
    results_df : pd.DataFrame
        Comparison results
    output_dir : str, default="tra_results"
        Output directory for artifacts
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "tra_model.joblib")
    tra_model.save_model(model_path)
    
    # Save results
    results_path = os.path.join(output_dir, "comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Save performance report
    report = generate_performance_report(tra_model, results_df)
    report_path = os.path.join(output_dir, "performance_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save TRA-specific report if available
    if hasattr(tra_model, 'get_performance_report'):
        tra_report = tra_model.get_performance_report()
        tra_report_path = os.path.join(output_dir, "tra_detailed_report.txt")
        with open(tra_report_path, 'w') as f:
            f.write(tra_report)
    
    print(f"Model artifacts saved to {output_dir}/")
    print(f"- Model: {model_path}")
    print(f"- Results: {results_path}")
    print(f"- Report: {report_path}")


def validate_input(X, y):
    """Validate input arrays for X and y."""
    if X is None or y is None:
        raise ValueError("X and y cannot be None")
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    return X, y

def format_performance_metrics(metrics):
    """Format performance metrics dictionary as a string."""
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.3f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)

if __name__ == "__main__":
    print("This is utils.py. Add your test or demo code here if needed.")