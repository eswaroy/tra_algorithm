# TRA Algorithm - Track/Rail Algorithm

[![PyPI version](https://badge.fury.io/py/tra-algorithm.svg)](https://badge.fury.io/py/tra-algorithm)
[![Python versions](https://img.shields.io/pypi/pyversions/tra-algorithm.svg)](https://pypi.org/project/tra-algorithm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/yourusername/tra-algorithm/workflows/CI/badge.svg)](https://github.com/yourusername/tra-algorithm/actions)
[![Coverage Status](https://codecov.io/gh/yourusername/tra-algorithm/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/tra-algorithm)

## Overview

The **Track/Rail Algorithm (TRA)** is a novel ensemble machine learning method that dynamically routes data through specialized "tracks" based on signal conditions. Unlike traditional ensemble methods that combine predictions, TRA creates multiple specialized models (tracks) and intelligently switches between them during prediction based on real-time signal evaluation.

## Key Features

- 🚄 **Dynamic Track Switching**: Intelligent routing of data through specialized models
- ⚡ **Parallel Processing**: Optimized signal evaluation with concurrent processing
- 🎯 **Adaptive Learning**: Self-optimizing parameters based on performance feedback
- 🔧 **Memory Optimization**: Automatic pruning of underused tracks
- 📊 **Rich Visualization**: Comprehensive model structure and performance visualization
- 🧪 **Dual Task Support**: Both classification and regression tasks
- 📈 **Performance Monitoring**: Detailed statistics and reporting

## Installation

Install TRA Algorithm using pip:

```bash
pip install tra-algorithm
```

For development installation:

```bash
git clone https://github.com/yourusername/tra-algorithm.git
cd tra-algorithm
pip install -e ".[dev]"
```

## Quick Start

### Classification Example

```python
from tra_algorithm import OptimizedTRA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train TRA
tra = OptimizedTRA(
    task_type="classification",
    n_tracks=5,
    random_state=42,
    parallel_signals=True,
    enable_track_pruning=True
)

tra.fit(X_train, y_train)

# Make predictions
y_pred = tra.predict(X_test)
y_proba = tra.predict_proba(X_test)

# Evaluate performance
accuracy = tra.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Regression Example

```python
from tra_algorithm import OptimizedTRA
from sklearn.datasets import make_regression

# Create sample data
X, y = make_regression(n_samples=1000, n_features=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train TRA for regression
tra = OptimizedTRA(
    task_type="regression",
    n_tracks=4,
    signal_threshold=0.15,
    feature_selection=True
)

tra.fit(X_train, y_train)
y_pred = tra.predict(X_test)

# Get performance metrics
mse_score = -tra.score(X_test, y_test)  # Negative MSE
print(f"MSE: {mse_score:.4f}")
```

## Advanced Features

### Model Visualization

```python
# Visualize the TRA structure
tra.visualize("tra_structure.png", figsize=(12, 8))

# Get detailed performance report
print(tra.get_performance_report())

# Get track statistics
stats = tra.get_track_statistics()
```

### Parameter Optimization

```python
# Optimize parameters using validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
tra.fit(X_train, y_train)
optimization_results = tra.optimize_parameters(X_val, y_val)
```

### Model Persistence

```python
# Save and load models
tra.save_model("my_tra_model.joblib")
loaded_tra = OptimizedTRA.load_model("my_tra_model.joblib")
```

## Algorithm Details

### How TRA Works

1. **Track Creation**: Multiple specialized models (tracks) are trained on different bootstrap samples
2. **Signal Generation**: Signals are created between tracks to detect when switching is beneficial
3. **Dynamic Routing**: During prediction, data is routed through tracks based on signal evaluation
4. **Performance Optimization**: Tracks and signals are continuously monitored and optimized

### Key Components

- **Tracks**: Specialized models trained on different data subsets
- **Signals**: Conditions that trigger switching between tracks
- **Records**: Individual data points with routing history
- **Enhanced Signal Conditions**: Advanced switching logic with regression optimization

## Parameters

### Main Parameters

- `task_type`: "classification" or "regression"
- `n_tracks`: Number of specialized tracks to create (default: 3)
- `signal_threshold`: Threshold for track switching (default: 0.1)
- `parallel_signals`: Enable parallel signal evaluation (default: True)
- `enable_track_pruning`: Enable automatic track pruning (default: True)
- `feature_selection`: Enable automatic feature selection (default: True)
- `handle_imbalanced`: Handle class imbalance (classification only, default: True)

### Performance Parameters

- `n_estimators`: Number of estimators per track (default: 50)
- `max_depth`: Maximum depth of track estimators (default: 6)
- `max_workers`: Maximum parallel workers (default: 4)
- `pruning_interval`: Interval for track pruning (default: 100)

## Performance Comparison

TRA has been tested against standard ensemble methods and shows competitive performance with additional benefits:

- **Adaptability**: Dynamically adjusts to data patterns
- **Interpretability**: Clear visualization of decision paths
- **Efficiency**: Optimized memory usage through track pruning
- **Robustness**: Handles both classification and regression effectively

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.3.0
- joblib >= 1.0.0
- networkx >= 2.6.0 (for visualization)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tra_algorithm --cov-report=html

# Run specific test file
pytest tests/test_core.py
```

## Documentation

Detailed documentation is available in the `docs/` directory. Build documentation locally:

```bash
cd docs
make html
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a history of changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TRA Algorithm in your research, please cite:

```bibtex
@software{tra_algorithm,
  title={TRA Algorithm: Track/Rail Algorithm for Dynamic Ensemble Learning},
  author={TRA Algorithm Team},
  year={2024},
  url={https://github.com/yourusername/tra-algorithm}
}
```

## Support

- 📧 Email: contact@tra-algorithm.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/tra-algorithm/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/tra-algorithm/discussions)

## Acknowledgments

- Built on top of scikit-learn
- Inspired by ensemble learning research
- Thanks to all contributors and users

---

**Made with ❤️ by the TRA Algorithm Team**