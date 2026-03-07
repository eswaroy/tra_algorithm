
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, make_friedman1, fetch_california_housing, load_diabetes
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error, r2_score
import time
import warnings
from tra_algorithm.core import OptimizedTRA

# Suppress warnings
warnings.filterwarnings('ignore')

# Try importing boosting libraries
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

def get_models(task_type):
    models = {}
    
    # Custom Model
    if task_type == 'classification':
        models['TRA'] = OptimizedTRA(task_type='classification', n_tracks=3, random_state=42)
        models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        models['SVM'] = SVC(probability=True, random_state=42)
        if XGB_AVAILABLE:
            models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        if LGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)
        if CAT_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(verbose=0, random_state=42)
            
    elif task_type == 'regression':
        models['TRA'] = OptimizedTRA(task_type='regression', n_tracks=3, random_state=42)
        models['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        models['SVR'] = SVR()
        if XGB_AVAILABLE:
            models['XGBoost'] = XGBRegressor(random_state=42)
        if LGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(random_state=42, verbose=-1)
        if CAT_AVAILABLE:
            models['CatBoost'] = CatBoostRegressor(verbose=0, random_state=42)
            
    return models

def run_benchmark():
    results = []
    
    print("Starting Benchmark...")
    print("-" * 50)
    
    # Define Datasets
    datasets = [
        ('Breast Cancer', 'classification', load_breast_cancer),
        ('Wine', 'classification', load_wine),
        ('Digits', 'classification', load_digits),
        ('Diabetes', 'regression', load_diabetes),
        ('Friedman1', 'regression', lambda: make_friedman1(n_samples=1000, n_features=10, noise=0.1, random_state=42)),
        # California Housing is large, might take time, but good for regression
        ('California Housing', 'regression', fetch_california_housing) 
    ]
    
    for name, task, loader in datasets:
        print(f"\nProcessing Dataset: {name} ({task})")
        
        try:
            data = loader()
            if isinstance(data, tuple):
                X, y = data
            else:
                X, y = data.data, data.target
            
            # Subsample large datasets for speed if needed (California Housing has ~20k samples)
            if X.shape[0] > 5000:
                print(f"  Subsampling from {X.shape[0]} to 5000 samples...")
                indices = np.random.choice(X.shape[0], 5000, replace=False)
                X = X[indices]
                y = y[indices]

            models = get_models(task)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task == 'classification' else KFold(n_splits=5, shuffle=True, random_state=42)
            
            for model_name, model in models.items():
                print(f"  Training {model_name}...", end='', flush=True)
                
                scoring = {
                    'classification': {'accuracy': 'accuracy', 'f1': 'f1_weighted'},
                    'regression': {'neg_rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
                }[task]
                
                start_time = time.time()
                try:
                    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1) # n_jobs=1 to avoid concurrency issues with custom model parallelism
                    fit_time = time.time() - start_time
                    
                    if task == 'classification':
                        res = {
                            'Dataset': name,
                            'Task': task,
                            'Model': model_name,
                            'Main_Metric': np.mean(scores['test_accuracy']),
                            'Main_Metric_Name': 'Accuracy',
                            'Secondary_Metric': np.mean(scores['test_f1']),
                            'Secondary_Metric_Name': 'F1 Score',
                            'Time (s)': fit_time
                        }
                    else:
                        res = {
                            'Dataset': name,
                            'Task': task,
                            'Model': model_name,
                            'Main_Metric': -np.mean(scores['test_neg_rmse']), # Convert neg_rmse to positive RMSE
                            'Main_Metric_Name': 'RMSE (Lower is Better)',
                            'Secondary_Metric': np.mean(scores['test_r2']),
                            'Secondary_Metric_Name': 'R2 Score',
                            'Time (s)': fit_time
                        }
                    results.append(res)
                    print(f" Done. ({res['Main_Metric']:.4f})")
                    
                except Exception as e:
                    print(f" Failed: {str(e)}")
        
        except Exception as e:
            print(f"Error processing dataset {name}: {str(e)}")

    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv('benchmark_results.csv', index=False)
    print("\nBenchmark Complete. Results saved to benchmark_results.csv")
    
    # Generate Summary Report
    with open('benchmark_summary.txt', 'w') as f:
        f.write("Benchmark Summary\n")
        f.write("="*50 + "\n\n")
        for dataset in df_results['Dataset'].unique():
            f.write(f"Dataset: {dataset}\n")
            subset = df_results[df_results['Dataset'] == dataset].sort_values(by='Main_Metric', ascending=(df_results[df_results['Dataset'] == dataset].iloc[0]['Task'] == 'regression'))
            f.write(subset.to_string(index=False))
            f.write("\n\n" + "-"*50 + "\n\n")

if __name__ == "__main__":
    run_benchmark()
