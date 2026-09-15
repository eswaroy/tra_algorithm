```markdown
# Changelog

All notable changes to the TRA Algorithm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.6] - 2026-09-15

### Fixed
- `check_is_fitted(self, 'fitted_')` was a silent no-op: it checked only for the *presence* of
  the `fitted_` attribute (always present after `__init__`, regardless of fit state), not
  whether the model was actually fitted. An unfitted estimator's `predict()`/`predict_proba()`
  would proceed instead of raising `NotFittedError`, eventually surfacing as a confusing,
  unrelated `ValueError`. Added `__sklearn_is_fitted__()` so `check_is_fitted(self)` now
  correctly reports fit state.
- Router/regional-track label alignment: when a router or regional sub-router was trained on a
  sparse subset of classes, its `predict_proba()` column positions did not necessarily match
  global track/class indices by position. Fixed to map through the router's own `classes_`
  instead of assuming positional alignment.
- OOF (out-of-fold) surrogate fidelity: OOF cross-validation for stacking/router/correction-track
  training now clones each *actual* deployed track's classifier (preserving model type,
  hyperparameters, and calibration wrapper) instead of an unrelated generic template, so OOF
  quantities describe the experts that are actually deployed at inference time.
  - `StratifiedKFold` fold count is now capped by the rarest class's sample count, preventing
    folds with missing classes during OOF computation.
  - A failed OOF cell now raises instead of silently filling in as an all-zero prediction, which
    previously could masquerade as a valid (wrong) probability distribution downstream.
- Conformal prediction intervals: replaced an interpolated `np.quantile` call (which undercovers
  on small calibration sets) with the correct discrete order-statistic used in split-conformal
  prediction.
- `combination_mode="flat_average"` no longer wastes part of the training set on an unused
  router holdout split (it has no router to protect from leakage).
- Packaging: fixed `MANIFEST.in`'s `__pycache__` exclusion (was a no-op due to incorrect glob
  syntax), removed `traauto`/`build` from package discovery, and removed an unnecessary
  `setuptools_scm` build dependency whose default file-finder was silently including every
  git-tracked file (including large binaries) in source distributions regardless of
  `MANIFEST.in`.
- README: corrected a stale claim that TRA was not yet published on PyPI.

No public API changes; `EnhancedTRA`/`OptimizedTRA` constructor signatures, defaults, and
`fit`/`predict`/`predict_proba` behavior are unchanged for already-fitted, well-formed inputs.
Validated via the existing unit test suite plus a dedicated before/after comparison across
6 benchmark datasets in both recommended configurations (classification/stacking and
regression/routing): identical accuracy, log-loss, and RMSE, no regressions found.

## [1.0.5] - 2026-03-07

### Documentation
- Complete README rewrite documenting actual MoE architecture
- Added comprehensive architecture diagrams showing signal flow
- Documented all 11 integrated improvements with technical details
- Added parameter reference table (20+ parameters)
- Created advanced features section (hard/soft routing, temperature scaling, dynamic spawning)
- Added troubleshooting and model inspection guides
- Included streaming & out-of-core learning examples

### Accuracy Improvements
- README now accurately represents SignalExtractor with 5 structural signals
- Documented EnhancedTRA class features matching actual implementation
- Router backends (XGBoost, CatBoost, MLP, LightGBM) properly explained
- Correction track (TRA-Boost) functionality documented
- Confidence-based abstention feature documented

## [1.0.4] - 2026-03-07

### Added
- Stronger Router implementations (XGBoost, CatBoost, MLP, LightGBM)
- Heterogeneous Expert Tracks with specialized models
- Increased Number of Tracks (5-8+) for better specialization
- Load Balancing Loss for fair expert utilization
- Top-K Routing for selective expert usage
- Expert Capacity Control for improved memory efficiency
- Router Meta-Features for enhanced routing decisions
- Temperature-Scaled Soft Routing for better convergence
- Dynamic Track Creation capabilities
- Track Specialization via K-Means Clustering

### Improved
- Algorithm performance significantly enhanced with mixture-of-experts approach
- Switch Transformer-inspired routing mechanism
- Signal-Guided Routing with structural signal extraction
- Memory optimization and computational efficiency

## [1.0.0] - 2024-12-XX

### Added
- Initial release of the Track/Rail Algorithm (TRA)
- OptimizedTRA class with support for classification and regression
- Multi-track architecture with intelligent signal system
- Parallel processing capabilities for improved performance
- Automatic track pruning for memory optimization
- Parameter optimization functionality
- Comprehensive performance tracking and analytics
- Visualization capabilities with NetworkX integration
- Model persistence (save/load functionality)
- Extensive test suite with unit tests
- Documentation and examples
- Support for feature selection and class imbalance handling
- Integration with scikit-learn ecosystem

### Features
- **Multi-Track Learning**: Create multiple specialized models that focus on different aspects of the data
- **Intelligent Switching**: Automatic track switching based on prediction confidence
- **Performance Optimization**: Parallel signal evaluation and track pruning
- **Analytics**: Detailed performance reports and statistics
- **Visualization**: Network graphs showing track relationships and performance
- **Scikit-learn Compatible**: Follows scikit-learn API conventions

### Supported Algorithms
- Random Forest based tracks for both classification and regression
- Enhanced signal conditions with regression-specific optimizations
- StandardScaler for feature normalization
- SelectKBest for feature selection
- Class weight balancing for imbalanced datasets

### Requirements
- Python >= 3.7
- NumPy >= 1.19.0
- Pandas >= 1.2.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.3.0
- Joblib >= 1.0.0
- NetworkX >= 2.5 (optional, for visualization)

### Documentation
- Comprehensive API documentation
- Quick start guide with examples
- Advanced usage patterns
- Performance optimization tips
- Troubleshooting guide

### Testing
- Unit tests for all core functionality
- Integration tests for model persistence
- Performance benchmarking tests
- Edge case handling tests

## [Unreleased]

### Planned Features
- Support for additional base estimators (XGBoost, LightGBM)
- Advanced signal conditions (time-based, performance-based)
- Online learning capabilities
- GPU acceleration support
- Advanced visualization options
- Model interpretability features
- Hyperparameter optimization integration
- Streaming data support

### Known Issues
- Visualization requires NetworkX installation
- Large models may consume significant memory
- Parallel processing performance varies by system

---

## Version History Summary

- **1.0.0**: Initial release with core TRA functionality
- **Future releases**: Will include advanced features and optimizations based on user feedback

## Contributing

We welcome contributions! Please see our contributing guidelines for details on how to submit improvements, bug fixes, and new features.

## Support

For support, please:
1. Check the documentation and examples
2. Review known issues in this changelog
3. Submit issues on the project repository
4. Join our community discussions
```
