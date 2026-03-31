# TRA Algorithm - Enhanced Track/Rail Algorithm

[![PyPI version](https://badge.fury.io/py/tra-algorithm.svg)](https://badge.fury.io/py/tra-algorithm)
[![Python versions](https://img.shields.io/pypi/pyversions/tra-algorithm.svg)](https://pypi.org/project/tra-algorithm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The **Enhanced Track/Rail Algorithm (TRA)** is a sophisticated **Mixture-of-Experts (MoE)** ensemble machine learning architecture that combines Switch Transformer-inspired routing with signal-guided expert gating. Unlike traditional ensemble methods that combine predictions uniformly, TRA intelligently routes data to specialized expert tracks based on both input features AND structural signals about data difficulty, density, and anomaly scores.

**Core Innovation**: Signal-Guided Routing extracts 5 structural signals (expert disagreement, prediction entropy, feature density, cluster distance, outlier score) to guide MoE routing for improved specialization and reduced expert collapse.

## Key Features

- 🏗️ **Mixture-of-Experts Architecture**: 5-8+ heterogeneous expert tracks (RF, LightGBM, XGBoost, SVM, MLP)
- 🚦 **Signal-Guided Routing**: Structural signal extraction for intelligent expert selection
- 🤖 **Stronger Router Models**: XGBoost, CatBoost, MLP, or LightGBM with meta-features
- 🔄 **Soft & Hard Routing Modes**: Temperature-scaled soft routing with weighted averaging
- ⚖️ **Load Balancing**: Prevents expert collapse via load balancing loss
- 📊 **Top-K Routing**: Route to multiple experts with confidence-weighted averaging
- 💾 **Expert Capacity Control**: Limit samples per expert for fairness and efficiency
- 🌱 **Dynamic Track Spawning**: Automatically create specialists for uncertain regions
- 🎯 **Track Specialization**: KMeans clustering for region-based expert specialization
- 📈 **Residual Correction**: TRA-Boost correction track for systematic error reduction
- 🔄 **Streaming Support**: Out-of-core learning with partial_fit() for incremental training
- 🧹 **Automatic Track Pruning**: Remove underused tracks for memory optimization
- 🛑 **Confidence-Based Abstention**: Option to abstain on low-confidence predictions
- 🧪 **Dual Task Support**: Both classification and regression tasks
- ⚡ **Parallel Processing**: Multi-threaded track predictions with ThreadPoolExecutor

## Installation

Install TRA Algorithm using pip:

```bash
pip install tra-algorithm
```

For development installation:

```bash
git clone https://github.com/eswaroy/tra_algorithm.git
cd tra_algorithm
pip install -e ".[dev]"
```

## Quick Start

### ⚠️ Before You Start: Key Pitfalls to Avoid

The most common mistakes that cause poor TRA performance:

1. **❌ Too Few Features After Selection** - Default feature selection was too aggressive (1/3 rule)
   - ✅ **FIX**: Use adaptive 60% retention (automatic, already fixed in v1.0.4+)

2. **❌ Mismatched Router & Track Models** - Using weak routers (decision trees) with strong tracks
   - ✅ **FIX**: Always use stronger routers (XGBoost/CatBoost/LightGBM) for optimal routing decisions

3. **❌ Insufficient Data for K-Fold Validation** - TRA needs 50+ samples per fold minimum
   - ✅ **FIX**: Use datasets with 500+ samples; smaller datasets use 3-5 fold CV instead of 10

4. **❌ Wrong Router Type for Task** - Using XGBoost router for regression
   - ✅ **FIX**: Use `router_type="lightgbm"` for regression, `router_type="catboost"` for classification

5. **❌ Missing Preprocessing** - Feeding raw unscaled features
   - ✅ **FIX**: Always enable `feature_selection=True` and let TRA handle preprocessing

6. **❌ Weak Cooling Parameter** - Using too high temperature for routing
   - ✅ **FIX**: Use `routing_temperature=0.8` for classification, `routing_temperature=1.0` for regression

7. **❌ Too Few or Too Many Tracks** - Using only 2 tracks or 20+ tracks
   - ✅ **FIX**: Use `n_tracks=5-7` for balanced expertise/computation tradeoff

8. **❌ Not Using Top-K Routing** - Hard routing to single expert defeats MoE ensemble benefit
   - ✅ **FIX**: Always use `top_k=2-3` with soft routing for better predictions

---

### 📊 Optimal Configuration by Dataset Size & Complexity

#### **Small Datasets (100-500 samples)**
For limited data, use fewer tracks and emphasize regularization:

```python
from tra_algorithm import OptimizedTRA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Small dataset
X, y = make_classification(n_samples=300, n_features=12, n_classes=2, 
                           n_informative=8, n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lightweight TRA for small data
tra_small = OptimizedTRA(
    task_type="classification",
    n_tracks=3,                          # Fewer tracks to avoid overfitting
    router_type="mlp",                   # Fast MLP router
    routing_mode="soft",
    routing_temperature=0.8,             # Sharp routing (conservative)
    top_k=2,                             # Ensemble of 2 experts
    use_meta_features=True,
    cluster_experts=False,               # Bootstrap instead of clustering for small data
    enable_correction_track=False,       # Skip correction track for small data
    feature_selection=True,
    n_estimators=30,                     # Smaller trees
    random_state=42
)

tra_small.fit(X_train, y_train)
print(f"Accuracy: {tra_small.score(X_test, y_test):.4f}")

# Use cross-validation for reliable estimates on small data
cv_scores = cross_val_score(tra_small, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

#### **Medium Datasets (500-5000 samples)**
Balanced approach - this is where TRA shines:

```python
from tra_algorithm import OptimizedTRA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Medium dataset - OPTIMAL CONFIG
X, y = make_classification(n_samples=2000, n_features=25, n_classes=3,
                           n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **OPTIMAL FOR MEDIUM DATA - This is the sweet spot for TRA**
tra_optimal = OptimizedTRA(
    task_type="classification",
    n_tracks=6,                          # 🎯 Optimal: 5-7 tracks
    router_type="catboost",              # 🎯 Best classification router
    routing_mode="soft",                 # 🎯 Soft routing for ensemble
    routing_temperature=0.8,             # 🎯 Sharp + smart routing
    top_k=2,                             # 🎯 Multi-expert routing
    use_meta_features=True,              # 🎯 Enable signal-guided routing
    cluster_experts=True,                # 🎯 KMeans specialization
    load_balance_strength=0.01,          # 🎯 Prevent expert collapse
    enable_correction_track=True,        # 🎯 TRA-Boost for error correction
    enable_track_pruning=True,           # 🎯 Auto-prune underused tracks
    confidence_spawn_threshold=0.3,      # 🎯 Spawn specialists for uncertain regions
    max_dynamic_tracks=2,                # 🎯 Limited dynamic growth
    feature_selection=True,              # 🎯 Adaptive 60% retention
    handle_imbalanced=True,              # 🎯 Class weight balancing
    n_estimators=75,                     # 🎯 Medium-strength trees
    max_depth=7,                         # 🎯 Reasonable tree depth
    random_state=42
)

tra_optimal.fit(X_train, y_train)

# Evaluation with all metrics
y_pred = tra_optimal.predict(X_test)
y_proba = tra_optimal.predict_proba(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'):.4f}")

# Inspect routing behavior
print(f"\nEnsemble Structure:")
print(f"  Active tracks: {len(tra_optimal.tracks)}")
print(f"  Router type: {tra_optimal.router_type}")
print(f"  Routing mode: {tra_optimal.routing_mode}")
```

---

#### **Large Datasets (5000+ samples)**
Maximize specialization and efficiency:

```python
from tra_algorithm import OptimizedTRA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Large dataset
X, y = make_classification(n_samples=10000, n_features=50, n_classes=5,
                           n_informative=30, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Full-power TRA for large data
tra_large = OptimizedTRA(
    task_type="classification",
    n_tracks=7,                          # More tracks for specialization
    router_type="xgboost",               # Fast XGBoost router for large data
    routing_mode="soft",
    routing_temperature=0.7,             # Even sharper routing
    top_k=3,                             # More ensemble depth
    use_meta_features=True,
    cluster_experts=True,                # Aggressive region-based specialization
    load_balance_strength=0.02,
    enable_correction_track=True,
    enable_track_pruning=True,
    confidence_spawn_threshold=0.2,      # More aggressive spawning
    max_dynamic_tracks=3,
    feature_selection=True,
    n_estimators=100,                    # Stronger base estimators
    max_depth=8,
    random_state=42
)

tra_large.fit(X_train, y_train)
print(f"Accuracy: {tra_large.score(X_test, y_test):.4f}")
```

---

### 🎯 Regression with Optimal Routing

```python
from tra_algorithm import OptimizedTRA
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(n_samples=2000, n_features=30, n_informative=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **OPTIMAL FOR REGRESSION**
tra_regression = OptimizedTRA(
    task_type="regression",
    n_tracks=6,
    router_type="lightgbm",              # 🎯 LightGBM is best for regression routing
    routing_mode="soft",
    routing_temperature=1.0,             # Smooth for regression (not sharp)
    top_k=2,                             # Ensemble averaging helps regression
    use_meta_features=True,
    cluster_experts=True,
    enable_correction_track=True,        # 🎯 TRA-Boost crucial for regression
    enable_track_pruning=True,
    feature_selection=True,
    n_estimators=80,
    max_depth=8,                         # Deeper trees for regression
    random_state=42
)

tra_regression.fit(X_train, y_train)
y_pred = tra_regression.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
```

---

### 🔄 Streaming/Online Learning with Incremental Updates

```python
from tra_algorithm import OptimizedTRA
import numpy as np

# Initial model on first batch
X_batch1, y_batch1 = make_classification(n_samples=500, n_features=20, random_state=1)
tra_stream = OptimizedTRA(
    task_type="classification",
    n_tracks=4,
    router_type="catboost",
    enable_track_pruning=True,
    random_state=42
)
tra_stream.fit(X_batch1, y_batch1)
print(f"Batch 1: Accuracy = {tra_stream.score(X_batch1, y_batch1):.4f}")

# Second batch - adapt to new data pattern
X_batch2, y_batch2 = make_classification(n_samples=500, n_features=20, random_state=2)
tra_stream.partial_fit(X_batch2, y_batch2)  # 🎯 Incremental learning
print(f"Batch 1+2: Accuracy = {tra_stream.score(X_batch2, y_batch2):.4f}")

# Third batch - TRA learns concept drift
X_batch3, y_batch3 = make_classification(n_samples=500, n_features=20, random_state=3)
tra_stream.partial_fit(X_batch3, y_batch3)
print(f"Batch 1+2+3: Accuracy = {tra_stream.score(X_batch3, y_batch3):.4f}")

print(f"\nTotal active tracks after streaming: {len(tra_stream.tracks)}")
```

---

### 🎛️ Advanced: Tuning for Maximum Performance

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from tra_algorithm import OptimizedTRA

# Grid search for optimal parameters
param_grid = {
    'n_tracks': [4, 5, 6, 7],
    'router_type': ['catboost', 'lightgbm'],
    'routing_temperature': [0.7, 0.8, 0.9],
    'top_k': [2, 3],
    'cluster_experts': [True, False]
}

# Note: GridSearchCV works with TRA since it implements sklearn API
tra_base = OptimizedTRA(task_type="classification", random_state=42)

grid_search = GridSearchCV(
    tra_base,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=2,  # Parallel across CV folds
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"\n🏆 Best Parameters: {grid_search.best_params_}")
print(f"🏆 Best CV Score: {grid_search.best_score_:.4f}")

best_tra = grid_search.best_estimator_
print(f"Test Set Accuracy: {best_tra.score(X_test, y_test):.4f}")
```

---

### 📈 Performance Monitoring & Diagnostics

```python
# Track-level performance inspection
print("Expert Track Performance Analysis:")
for track_name, track in tra_optimal.tracks.items():
    print(f"\n{track_name}:")
    print(f"  Performance Score: {track.performance_score:.3f}")
    print(f"  Usage Count: {track.usage_count}")
    print(f"  Capacity Violations: {track.capacity_violations}")
    print(f"  Avg Prediction Time: {track.get_average_prediction_time():.4f}s")

# Router meta-information
if tra_optimal.router_ is not None:
    print(f"\nRouter Information:")
    print(f"  Model Type: {type(tra_optimal.router_).__name__}")
    print(f"  Uses Meta-Features: {tra_optimal._router_uses_meta_features}")
    print(f"  Meta-Feature Dimension: {X_train.shape[1] + 5 if tra_optimal._router_uses_meta_features else X_train.shape[1]}")
```

---

### 💾 Save & Load Trained Models

```python
# Save to disk
tra_optimal.save_model("best_tra_model.joblib")
print("Model saved to 'best_tra_model.joblib'")

# Load from disk
from tra_algorithm import OptimizedTRA
loaded_tra = OptimizedTRA.load_model("best_tra_model.joblib")

# Use loaded model
y_pred = loaded_tra.predict(X_test)
print(f"Loaded Model Accuracy: {loaded_tra.score(X_test, y_test):.4f}")
```

---

### 🐛 Debugging Poor Performance

If TRA underperforms baseline models, check these:

**1. Check if preprocessing was skipped:**
```python
# ❌ DON'T DO THIS - raw unscaled features
tra = OptimizedTRA()
tra.fit(X_raw, y)  # Features not scaled!

# ✅ DO THIS - TRA handles preprocessing internally
tra = OptimizedTRA(feature_selection=True)  # Auto-scaling + feature selection
tra.fit(X_raw, y)
```

**2. Check router quality:**
```python
# If router is weak (MLP on huge data), switch to XGBoost/CatBoost
if X_train.shape[0] > 5000 and X_train.shape[1] > 50:
    router_type = "xgboost"  # Fast + accurate
else:
    router_type = "catboost"  # More accurate
```

**3. Verify routing confidence:**
```python
from tra_algorithm import OptimizedTRA

y_proba = tra_optimal.predict_proba(X_test)
routing_confidence = y_proba.max(axis=1)

# If mean confidence < 0.5, routing is uncertain → increase n_tracks
print(f"Mean Routing Confidence: {routing_confidence.mean():.3f}")
if routing_confidence.mean() < 0.5:
    print("⚠️  Low routing confidence → try n_tracks=7-8 or enable cluster_experts=True")
```

**4. Check for overfitting:**
```python
# Compare train vs test scores
train_score = tra_optimal.score(X_train, y_train)
test_score = tra_optimal.score(X_test, y_test)

print(f"Train Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"Overfitting Gap: {(train_score - test_score):.4f}")

# If gap > 0.05:
# → Reduce n_estimators or max_depth
# → Increase feature_selection (already at 60%)
# → Enable load_balance_strength=0.02
```

---

### ✅ Quick Performance Checklist

```
Classification Task Checklist:
☑ n_tracks = 5-7 (never < 3 or > 10)
☑ router_type = "catboost" (best accuracy) or "xgboost" (fast)
☑ routing_mode = "soft" (ensemble is better)
☑ routing_temperature = 0.7-0.9 (sharp routing)
☑ top_k = 2-3 (multi-expert blending)
☑ use_meta_features = True (signal-guided routing)
☑ cluster_experts = True (region specialization)
☑ enable_correction_track = True (error correction)
☑ feature_selection = True (adaptive 60%)
☑ n_estimators = 50-100 (data-dependent)

Regression Task Checklist:
☑ n_tracks = 6-7
☑ router_type = "lightgbm" (best for regression) 
☑ routing_mode = "soft"
☑ routing_temperature = 1.0 (smooth routing)
☑ top_k = 2-3
☑ use_meta_features = True
☑ cluster_experts = True
☑ enable_correction_track = True ⚠️ CRITICAL FOR REGRESSION
☑ feature_selection = True
☑ max_depth = 7-8 (deeper for regression)
```

## Advanced Features

### 🎯 Hard vs. Soft Routing: Which Should You Use?

**Hard Routing** selects a single best expert - simpler, faster, but less robust:
```python
# Hard Routing: Always picks single best expert (like hard gating in MoE)
tra_hard = OptimizedTRA(
    routing_mode="hard",
    n_tracks=5,
    router_type="xgboost"
)
tra_hard.fit(X_train, y_train)
# Each sample routed to exactly 1 expert
y_pred = tra_hard.predict(X_test)
```

**🏆 Soft Routing** - RECOMMENDED for best performance:
```python
# Soft Routing: Blends predictions from top-K experts (ensemble method)
# ✅ Recommended: Provides robustness + higher accuracy
tra_soft = OptimizedTRA(
    routing_mode="soft",         # Enable weighted ensemble
    routing_temperature=0.8,     # Control routing sharpness
    top_k=2,                     # Use top 2 experts
    n_tracks=6
)
tra_soft.fit(X_train, y_train)
# Each sample uses weighted averaging from K experts
y_pred = tra_soft.predict(X_test)
```

**When to use each:**
- Use **Hard Routing** when: prediction speed is critical (low-latency inference), or model interpretability needed
- Use **Soft Routing** when: accuracy is priority (recommended for most cases), or ensemble diversity helps

---

### 🌡️ Temperature Scaling: Sharp vs. Smooth Routing

```python
# Sharp routing (temperature < 1.0)
# - More decisive expert selection
# - Better for confident predictions
# - Can miss borderline/ambiguous samples
tra_sharp = OptimizedTRA(
    routing_temperature=0.5,     # Sharp routing
    routing_mode="soft",
    top_k=1  # Even with soft mode, sharp temp makes one expert dominate
)
tra_sharp.fit(X_train, y_train)

# Smooth routing (temperature > 1.0)
# - More uniform expert usage
# - Better for ambiguous/borderline samples
# - Can dilute strong experts' decisions
tra_smooth = OptimizedTRA(
    routing_temperature=2.0,     # Smooth routing
    routing_mode="soft",
    top_k=3  # More balanced blending across all experts
)
tra_smooth.fit(X_train, y_train)

# Optimal middle ground (temperature=0.8-1.0)
tra_optimal = OptimizedTRA(
    routing_temperature=0.8,     # Sweet spot
    routing_mode="soft",
    top_k=2
)
tra_optimal.fit(X_train, y_train)

# Diagnostic: Check routing confidence
y_proba = tra_optimal.predict_proba(X_test)
mean_confidence = y_proba.max(axis=1).mean()
print(f"Mean Routing Confidence: {mean_confidence:.3f}")
# If < 0.5: routing uncertain, try lower temperature
# If > 0.95: routing overconfident, try higher temperature
```

**Temperature Guidelines:**
| Temperature | Behavior | Use Case |
|---|---|---|
| 0.5 | Sharp, decisive | Confident predictions; reject uncertain |
| 0.8 | Medium-sharp ⭐ | **Default for classification** |
| 1.0 | Balanced | **Default for regression** |
| 1.5 | Medium-smooth | Mixed confidence levels |
| 2.0+ | Smooth, uniform | Very ambiguous data |

---

### 🚀 Top-K Routing: Single vs. Multi-Expert Ensemble

```python
# top_k=1 (No ensemble, pick best expert)
tra_single = OptimizedTRA(
    routing_mode="soft",
    top_k=1  # Only use best expert's prediction
)
# Fast inference, but loses ensemble benefit

# top_k=2 (Blend 2 experts) - RECOMMENDED
tra_pair = OptimizedTRA(
    routing_mode="soft",
    top_k=2  # ⭐ Good balance of speed & accuracy
)
# Good ensemble diversity with minimal overhead

# top_k=3 (Blend 3 experts)
tra_triple = OptimizedTRA(
    routing_mode="soft",
    top_k=3  # More robust, slightly slower
)
# Higher accuracy but lower inference speed

# top_k=n_tracks (Use all experts)
tra_full = OptimizedTRA(
    routing_mode="soft",
    top_k=6,  # Use all 6 experts
    n_tracks=6
)
# Maximum ensemble power, slowest inference
# Rarely beneficial - better to keep strong routers

# Recommendation:
# Small data (< 500): top_k=2
# Medium data (500-5000): top_k=2-3
# Large data (> 5000): top_k=3, or top_k=all_tracks if router is weak
```

---

### 🌱 Dynamic Expert Spawning: Auto-Growing Mixture-of-Experts

```python
# Enable automatic specialist creation for uncertain regions
tra_dynamic = OptimizedTRA(
    task_type="classification",
    n_tracks=4,                          # Initial 4 experts
    confidence_spawn_threshold=0.3,      # 🎯 Key parameter!
    max_dynamic_tracks=3,                # Grow up to 4+3=7 total
    enable_track_pruning=True            # Clean up unused specialists
)

tra_dynamic.fit(X_train, y_train)
print(f"Initial tracks: 4")
print(f"Final tracks after spawning: {len(tra_dynamic.tracks)}")

# During prediction phase, if > 30% of samples have low confidence,
# TRA automatically creates a new specialist track trained on those hard samples

# Diagnostic: Check if spawning happened
print(f"Dynamic tracks created: {tra_dynamic._dynamic_tracks_created}")

# Recommendations:
# - classification: spawn_threshold = 0.2-0.4
# - regression: spawn_threshold = 0.25-0.35
# - Set max_dynamic_tracks = 2-3 to prevent uncontrolled growth
```

---

### 🧹 Confidence-Based Prediction Abstention

```python
# Refuse to predict on low-confidence samples
tra_abstain = OptimizedTRA(
    task_type="classification",
    abstention_threshold=0.5,      # Abstain if confidence < 50%
    abstention_class="UNCERTAIN"   # Return this for uncertain samples
)

tra_abstain.fit(X_train, y_train)
y_pred = tra_abstain.predict(X_test)

# Results will contain class predictions AND "UNCERTAIN" values
# This is useful for:
# - Human-in-the-loop systems (send uncertain to human review)
# - Safety-critical applications (only predict when confident)
# - Reducing wrong predictions on ambiguous boundaries

# Count abstentions
n_abstain = (y_pred == "UNCERTAIN").sum()
print(f"Abstained on {n_abstain}/{len(y_pred)} samples ({100*n_abstain/len(y_pred):.1f}%)")
```

---

### 🎯 KMeans Clustering vs. Bootstrap Sampling for Track Specialization

```python
# **BOOTSTRAP SAMPLING** (cluster_experts=False)
# Each track gets random samples with replacement
# ✅ Good for: Random diversity, faster training
tra_bootstrap = OptimizedTRA(
    n_tracks=5,
    cluster_experts=False  # Each track trains on bootstrap samples
)
tra_bootstrap.fit(X_train, y_train)
# Result: Tracks are diverse but undefined specialization

# **🏆 KMEANS CLUSTERING** (cluster_experts=True) - RECOMMENDED
# Each track specializes to a data region (KMeans cluster)
# ✅ Good for: Interpretable regions, better for structured data
tra_clustered = OptimizedTRA(
    n_tracks=5,
    cluster_experts=True  # ⭐ Track 0 = region 0, Track 1 = region 1, etc.
)
tra_clustered.fit(X_train, y_train)
# Result: Tracks assigned to specific input space regions

# When to use each:
# - Use Bootstrap: High-dimensional data, unstructured patterns
# - Use KMeans: Structured data with interpretable clusters
# - Use KMeans: User wants to understand track specialization
# - Use KMeans: Data has clear separation (e.g., multi-regime)
```

---

### 🔧 Custom Expert Track Models

```python
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC

# By default, TRA uses heterogeneous models: RF, LightGBM, XGBoost, SVM, MLP

# Create custom expert diversity
custom_experts = [
    GradientBoostingClassifier(n_estimators=100, max_depth=5),
    ExtraTreesClassifier(n_estimators=100, max_depth=7),
    RandomForestClassifier(n_estimators=100, max_depth=7),
    SVC(C=1.0, kernel='rbf', probability=True),  # Needs probability=True
    # Add more custom models as needed
]

tra_custom = OptimizedTRA(
    task_type="classification",
    track_models=custom_experts,  # Use your custom models
    router_type="catboost"
)
tra_custom.fit(X_train, y_train)

# Note: Custom models must have .fit(), .predict(), and 
# .predict_proba (for classification) or .predict (for regression)
```

---

### 📊 Model Inspection & Diagnostics

```python
# Get ensemble structure
print(f"Number of expert tracks: {len(tra_optimal.tracks)}")
print(f"Input features: {tra_optimal.n_features_in_}")
print(f"Classes: {tra_optimal.classes_}")
print(f"Router type: {tra_optimal.router_type}")
print(f"Routing mode: {tra_optimal.routing_mode}")
print(f"Task type: {tra_optimal.task_type}")

# Track-by-track performance
print("\n📈 Expert Track Performance Analysis:")
for track_name, track in tra_optimal.tracks.items():
    print(f"\n{track_name}:")
    print(f"  Performance Score: {track.performance_score:.3f}")
    print(f"  Usage Count: {track.usage_count}")
    print(f"  Capacity Violations: {track.capacity_violations}")
    print(f"  Avg Prediction Time: {track.get_average_prediction_time():.4f}s")

# Router information
if tra_optimal.router_ is not None:
    print(f"\n🎯 Router Information:")
    print(f"  Type: {type(tra_optimal.router_).__name__}")
    print(f"  Uses Meta-Features: {tra_optimal._router_uses_meta_features}")
    if tra_optimal._router_uses_meta_features:
        meta_dim = X_train.shape[1] + 5  # Original + 5 signals
        print(f"  Meta-Feature Dimension: {meta_dim}")

# Correction track info
if tra_optimal.correction_track_ is not None:
    print(f"\n📝 Correction Track (TRA-Boost):")
    print(f"  Type: {type(tra_optimal.correction_track_).__name__}")
    print(f"  Enabled: True")
else:
    print(f"\n📝 Correction Track: Disabled")
```

## Algorithm Details

### Architecture Overview

The Enhanced TRA implements a sophisticated Mixture-of-Experts (MoE) system with 11 integrated improvements:

```
Input Data
    ↓
Preprocessing (Scaling, Imputation, Handling Missing Values)
    ↓
Feature Selection (Adaptive 60% feature retention)
    ↓
Signal Extraction Layer (5 structural signals)
    ├→ Expert Disagreement (std of track predictions)
    ├→ Prediction Entropy (entropy of router probabilities)
    ├→ Feature Density Score (k-NN distance-based)
    ├→ Cluster Distance (KMeans centroid distance)
    └→ Outlier Score (IsolationForest anomaly detection)
    ↓
Stronger Router (XGBoost/CatBoost/MLP/LightGBM)
    ↓
Expert Tracks (Heterogeneous ensemble: RF, LightGBM, XGBoost, SVM, MLP)
    ├→ Track Specialization (KMeans clustering or bootstrap sampling)
    └→ Top-K Soft Routing (weighted averaging with temperature scaling)
    ↓
Correction Track (TRA-Boost): Residual error correction
    ↓
Final Prediction
```

### 11 Integrated Improvements

1. **Stronger Router**: Multiple backend options (XGBoost, CatBoost, MLP, LightGBM) instead of simple decision trees
2. **Heterogeneous Expert Tracks**: Diverse model types per track (RF, LightGBM, XGBoost, SVM, MLP) for diverse expertise
3. **Increased Tracks**: Support for 5-8+ expert tracks enabling fine-grained specialization
4. **Load Balancing Loss**: Prevents expert collapse and ensures balanced utilization across experts
5. **Top-K Routing**: Route to multiple experts with confidence-weighted averaging instead of hard expert selection
6. **Expert Capacity Control**: Limit samples per expert for fairness and memory efficiency
7. **Router Meta-Features**: Augment router input with track disagreement signals and structural signals
8. **Temperature-Scaled Soft Routing**: Smooth routing boundaries via temperature scaling (prevents sharp switches)
9. **Dynamic Track Spawning**: Automatically create specialized tracks for uncertain regions during inference
10. **Track Specialization via Clustering**: KMeans-based clustering assigns data regions to experts
11. **Signal-Guided Routing**: Structural signal extraction layer for awareness of data geometry and expert consensus

### How It Works

**Training Phase:**
1. Split data into 80% track training and 20% router holdout set
2. Create K heterogeneous expert tracks with bootstrap sampling or KMeans clustering
3. Extract structural signals (disagreement, entropy, density, cluster distance, outlier) from training data
4. Train the Stronger Router on holdout set to learn which expert is best per sample
5. Optionally train a residual correction track on misclassified samples (TRA-Boost)

**Prediction Phase:**
1. Preprocess input, extract features, compute structural signals
2. Use Stronger Router to get routing probabilities to each expert
3. **Hard Routing**: Select single best expert and use its prediction
4. **Soft Routing**: Weight all experts by router confidence, average predictions with temperature scaling
5. Apply correction track if available (especially for regression)
6. Confidence-based abstention if requested
7. Monitor for concept drift and dynamically spawn new specialists if needed

### Key Components

- **EnhancedTRA**: Main class implementing the Mixture-of-Experts algorithm
- **SignalExtractor**: Computes 5 structural signals for routing guidance
- **Track**: Individual expert track with performance monitoring and capacity control
- **Router**: Stronger routing model trained to select best experts

## Parameters Reference

### Router & Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | str | "classification" | "classification" or "regression" |
| `n_tracks` | int | 5 | Number of initial expert tracks |
| `max_tracks` | int | 8 | Maximum allowed expert tracks |
| `router_type` | str | "xgboost" | Router backend: "xgboost", "catboost", "mlp", "lightgbm" |
| `routing_mode` | str | "soft" | "hard" (single expert) or "soft" (weighted average) |
| `routing_temperature` | float | 1.0 | Temperature for soft routing (lower = sharper, higher = smoother) |
| `top_k` | int | 1 | Route to top-K experts (soft routing only) |

### Expert Track Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_models` | list | None | Custom model list for expert tracks |
| `cluster_experts` | bool | False | Use KMeans clustering for track specialization |
| `feature_selection` | bool | True | Enable adaptive feature selection (keeps 60% of features) |
| `n_estimators` | int | 50 | Trees per track estimator |
| `max_depth` | int | 6 | Max depth for tree-based tracks |
| `expert_capacity` | float | None | Samples per expert (auto-computed if None) |

### Enhancement Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_meta_features` | bool | True | Augment router input with track disagreement signals |
| `load_balance_strength` | float | 0.01 | Strength of load balancing loss |
| `enable_correction_track` | bool | True | Train TRA-Boost correction track |
| `enable_track_pruning` | bool | True | Automatically prune underused tracks |
| `confidence_spawn_threshold` | float | 0.3 | Trigger dynamic track spawning at this uncertainty ratio |
| `max_dynamic_tracks` | int | 3 | Maximum dynamically spawned tracks |

### Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `handle_imbalanced` | bool | True | Compute class weights for imbalanced data |
| `abstention_threshold` | float | 0.0 | Abstain when router confidence < threshold |
| `abstention_class` | any | None | Class/value to predict when abstaining |
| `random_state` | int | None | Random seed for reproducibility |
| `max_workers` | int | 4 | Max worker threads (capped at 8) |
| `pruning_interval` | int | 100 | Check track pruning every N predictions |

### Model Persistence & Inspection

```python
# Save trained model with all state
tra_optimal.save_model("optimal_tra_model.joblib")
print("✅ Model saved to 'optimal_tra_model.joblib'")

# Load model back and use immediately
from tra_algorithm import OptimizedTRA
loaded_tra = OptimizedTRA.load_model("optimal_tra_model.joblib")

# Make predictions with loaded model
y_pred = loaded_tra.predict(X_test)
accuracy = loaded_tra.score(X_test, y_test)
print(f"✅ Loaded Model Accuracy: {accuracy:.4f}")

# Basic model attributes
print(f"Task type: {loaded_tra.task_type}")
print(f"Number of tracks: {len(loaded_tra.tracks)}")
print(f"Input features: {loaded_tra.n_features_in_}")
print(f"Routing mode: {loaded_tra.routing_mode}")
```

## Performance & Benchmarking

The Enhanced TRA architecture provides several competitive advantages:

### When TRA Excels

- **High-Dimensional Data**: Adaptive feature selection (60% retention) handles dimensionality well
- **Multiple Regimes**: Different data distributions → heterogeneous experts specialize
- **Imbalanced Classes**: Class weight balancing + routing precision
- **Concept Drift**: Out-of-core learning with partial_fit() adapts to new patterns
- **Uncertain Regions**: Dynamic track spawning creates specialists for ambiguous boundaries
- **Regression with Outliers**: Correction track captures systematic residual patterns

### Computational Efficiency

- **Soft Routing**: Weighted average avoids all-or-nothing expert selection
- **Track Pruning**: Removes underused experts to reduce memory/computation
- **Parallel Processing**: ThreadPoolExecutor-based concurrent track predictions
- **Adaptive Features**: 60% feature retention reduces input dimensionality
- **Expert Capacity Control**: Prevents any single expert from becoming a bottleneck

## Requirements

```
Python >= 3.8
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.3.0
joblib >= 1.0.0
networkx >= 2.6.0 (optional, for visualization)
```

### Optional Dependencies

For advanced router models:
```bash
pip install xgboost catboost lightgbm
```

If any optional dependency is missing, TRA gracefully falls back to available implementations.

## Troubleshooting

### Router training takes long time
- Reduce `n_tracks` to 3-4 for faster training
- Use `router_type="mlp"` which trains faster than tree-based routers

### High memory usage
- Enable `enable_track_pruning=True` (default) to remove unused experts
- Use `cluster_experts=True` to specialize experts to specific data regions
- Reduce `n_estimators` per track

### Poor performance on new data (concept drift)
- Use `partial_fit()` to incrementally retrain on new batches
- Enable `confidence_spawn_threshold < 1.0` to automatically spawn specialists
- Increase `routing_temperature` for smoother routing decisions

### Soft routing predictions don't change much
- This is expected! Temperature scaling prevents sharp switches
- Lower `routing_temperature` for sharper expert selection
- Try `routing_mode="hard"` to use single expert selection

## Citation

If you use TRA Algorithm in your research or projects, please cite:

```bibtex
@software{tra_algorithm2025,
  title={Enhanced Track/Rail Algorithm: Mixture-of-Experts with Signal-Guided Routing},
  author={Ranga Eswar, Dasari},
  year={2025},
  url={https://github.com/eswaroy/tra_algorithm},
  note={Version 1.0.4+: 11 improvements integrated including MoE routing, signal-guided expertise, dynamic track spawning}
}
```

## Support & Contact

- 📧 **Email**: rangaeswar890@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/eswaroy/tra_algorithm/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/eswaroy/tra_algorithm/discussions)
- 📚 **Documentation**: See [docs/](docs/) and [CHANGELOG.md](CHANGELOG.md)
