"""
Unit tests for utility functions.
"""
import unittest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tra_algorithm.core import OptimizedTRA
from tra_algorithm.utils import validate_input, format_performance_metrics


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_validate_input(self):
        """Test input validation function."""
        # Valid inputs
        X_valid = np.array([[1, 2], [3, 4], [5, 6]])
        y_valid = np.array([0, 1, 0])
        
        X_validated, y_validated = validate_input(X_valid, y_valid)
        np.testing.assert_array_equal(X_validated, X_valid)
        np.testing.assert_array_equal(y_validated, y_valid)
        
        # Invalid inputs
        with self.assertRaises(ValueError):
            validate_input(None, y_valid)
        
        with self.assertRaises(ValueError):
            validate_input(X_valid, None)
        
        # Mismatched shapes
        X_mismatch = np.array([[1, 2]])
        with self.assertRaises(ValueError):
            validate_input(X_mismatch, y_valid)
    
    def test_format_performance_metrics(self):
        """Test performance metrics formatting."""
        metrics = {
            'accuracy': 0.8567,
            'f1_score': 0.7234,
            'precision': 0.9012
        }
        
        formatted = format_performance_metrics(metrics)
        self.assertIsInstance(formatted, str)
        self.assertIn('accuracy', formatted.lower())
        self.assertIn('0.857', formatted)  # Rounded to 3 decimal places
    
    def test_model_save_load_integration(self):
        """Test model save/load functionality."""
        # Create and train a simple model
        X, y = OptimizedTRA.create_example_dataset(
            task_type="classification", n_samples=50
        )
        
        tra = OptimizedTRA(task_type="classification", n_tracks=2, random_state=42)
        tra.fit(X, y)
        
        # Test saving and loading (Windows-safe)
        tmp = tempfile.NamedTemporaryFile(suffix='.joblib', delete=False)
        tmp_path = tmp.name
        tmp.close()  # Close so joblib can use it
        try:
            tra.save_model(tmp_path)
            loaded_tra = OptimizedTRA.load_model(tmp_path)
            
            # Test that loaded model works
            y_pred_original = tra.predict(X[:10])
            y_pred_loaded = loaded_tra.predict(X[:10])
            
            np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
            
        finally:
            os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()