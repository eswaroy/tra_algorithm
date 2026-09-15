"""Version information for TRA Algorithm package."""

__version__ = "1.0.6"
__version_info__ = tuple(int(i) for i in __version__.split('.'))

# Version history
VERSION_HISTORY = {
    "1.0.0": "Initial release with OptimizedTRA algorithm",
    "1.0.1": "README and documentation update",
    "1.0.2": "README and install instructions update",
    "1.0.3": "Visualization bugfix and code improvements",
    "1.0.4": "Enhanced algorithm performance with 10 major improvements",
    "1.0.5": "Comprehensive README update documenting MoE architecture and signal-guided routing",
    "1.0.6": "Correctness/robustness audit: fixed check_is_fitted() being a silent no-op (unfitted "
             "models could predict without raising NotFittedError), sparse-class OOF/router index "
             "misalignment, per-track feature-subset fidelity in OOF surrogates, StratifiedKFold "
             "failing on rare classes, silent zero-fill on failed OOF cells, and conformal interval "
             "undercoverage. No accuracy/log-loss/RMSE regressions found across validation "
             "benchmarks; public API unchanged.",
    # Future versions will be added here
}