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

### Classification with Advanced Routing

```python
from tra_algorithm import OptimizedTRA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Enhanced TRA with Mixture-of-Experts
tra = OptimizedTRA(
    task_type="classification",
    n_tracks=5,                          # 5 heterogeneous expert tracks
    router_type="xgboost",              # Stronger router model
    routing_mode="soft",                # Soft weighted routing
    routing_temperature=1.0,            # Temperature scaling
    top_k=2,                           # Route to top 2 experts
    use_meta_features=True,             # Use router meta-features
    cluster_experts=True,               # KMeans specialization
    enable_correction_track=True,       # TRA-Boost correction
    enable_track_pruning=True,          # Automatic pruning
    random_state=42
)

tra.fit(X_train, y_train)

# Predictions with MoE routing
y_pred = tra.predict(X_test)
y_proba = tra.predict_proba(X_test)

# Evaluate
accuracy = tra.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Regression with Signal-Guided Routing

```python
from tra_algorithm import OptimizedTRA
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create regression data
X, y = make_regression(n_samples=1000, n_features=15, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enhanced TRA for regression
tra = OptimizedTRA(
    task_type="regression",
    n_tracks=5,                         # 5 specialized expert tracks
    router_type="lightgbm",            # LightGBM router
    routing_mode="soft",                # Soft routing with weighted averaging
    feature_selection=True,             # Adaptive feature selection (keeps 60%)
    handle_imbalanced=False,            # Not applicable for regression
    random_state=42
)

tra.fit(X_train, y_train)
y_pred = tra.predict(X_test)

# Negative MSE as score
mse_score = -tra.score(X_test, y_test)
print(f"MSE: {mse_score:.4f}")
```

### Streaming Data with Out-of-Core Learning

```python
# Batch 1: Initial training
tra = OptimizedTRA(task_type="classification", n_tracks=3, random_state=42)
tra.fit(X_batch1, y_batch1)

# Batch 2: Incremental learning
tra.partial_fit(X_batch2, y_batch2)

# Batch 3: More data
tra.partial_fit(X_batch3, y_batch3)

# Trained router now evaluates both old and new tracks for concept drift
y_pred = tra.predict(X_test)
```

## Advanced Features

### Hard vs. Soft Routing

```python
# Hard Routing: Route to single best expert
tra = OptimizedTRA(routing_mode="hard", n_tracks=5)
tra.fit(X_train, y_train)
predictions = tra.predict(X_test)  # Single expert per sample

# Soft Routing: Weighted average across experts
tra = OptimizedTRA(routing_mode="soft", routing_temperature=1.0, top_k=3)
tra.fit(X_train, y_train)
predictions = tra.predict(X_test)  # Blended from top 3 experts
```

### Temperature Scaling for Routing

```python
# Lower temperature = sharper, more decisive routing
tra_sharp = OptimizedTRA(routing_temperature=0.5)  # Sharp expert selection

# Higher temperature = smoother, more uniform routing
tra_smooth = OptimizedTRA(routing_temperature=2.0)  # Blended predictions
```

### Dynamic Expert Spawning

```python
# Automatically create specialists for uncertain regions
tra = OptimizedTRA(
    confidence_spawn_threshold=0.3,  # Spawn if >30% predictions are uncertain
    max_dynamic_tracks=3              # Max 3 dynamically created tracks
)
tra.fit(X_train, y_train)
# During prediction, new experts appear for ambiguous samples
```

### Confidence-Based Prediction Abstention

```python
# Abstain (refuse to predict) on low-confidence cases
tra = OptimizedTRA(
    task_type="classification",
    abstention_threshold=0.5,   # Abstain if confidence < 50%
    abstention_class="UNKNOWN"  # Return "UNKNOWN" instead of prediction
)
tra.fit(X_train, y_train)
y_pred = tra.predict(X_test)
# Some predictions will be "UNKNOWN" for uncertain samples
```

### Expert Track Specialization

```python
# Cluster-based specialization: each expert learns a data region
tra_clustered = OptimizedTRA(cluster_experts=True, n_tracks=5)
tra_clustered.fit(X_train, y_train)  # Each track specializes to a KMeans cluster

# vs. Bootstrap-based: each expert gets random samples
tra_bootstrap = OptimizedTRA(cluster_experts=False, n_tracks=5)
tra_bootstrap.fit(X_train, y_train)  # Each track gets bootstrap resamples
```

### Custom Expert Track Models

```python
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

custom_models = [
    GradientBoostingClassifier(n_estimators=100),
    ExtraTreesClassifier(n_estimators=100),
    RandomForestClassifier(n_estimators=100),
]

tra = OptimizedTRA(task_type="classification", track_models=custom_models)
tra.fit(X_train, y_train)
```

### Model Inspection

```python
# Get track statistics
print(f"Number of expert tracks: {len(tra.tracks)}")
print(f"Router type: {tra.router_type}")
print(f"Routing mode: {tra.routing_mode}")

# Access individual tracks
for track_name, track in tra.tracks.items():
    print(f"{track_name}: {track.performance_score:.3f}")
    print(f"  Usage count: {track.usage_count}")
    print(f"  Avg prediction time: {track.get_average_prediction_time():.4f}s")
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

## Model Persistence

```python
# Save trained model
tra.save_model("my_tra_model.joblib")

# Load model (includes metadata)
loaded_tra = OptimizedTRA.load_model("my_tra_model.joblib")

# Access metadata
metadata = loaded_tra.metadata  # task_type, n_tracks, n_features, etc.
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