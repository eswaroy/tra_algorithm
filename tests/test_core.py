"""
Unit tests for TRA algorithm core functionality.
"""
import unittest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tra_algorithm.core import OptimizedTRA


class TestOptimizedTRA(unittest.TestCase):
    """Test cases for OptimizedTRA class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Classification dataset
        self.X_clf, self.y_clf = make_classification(
            n_samples=200, n_features=10, n_classes=3, n_informative=3, random_state=42
        )
        self.X_train_clf, self.X_test_clf, self.y_train_clf, self.y_test_clf = \
            train_test_split(self.X_clf, self.y_clf, test_size=0.3, random_state=42)
        
        # Regression dataset
        self.X_reg, self.y_reg = make_regression(
            n_samples=200, n_features=10, random_state=42
        )
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = \
            train_test_split(self.X_reg, self.y_reg, test_size=0.3, random_state=42)
    
    def test_classification_basic(self):
        """Test basic classification functionality."""
        tra = OptimizedTRA(task_type="classification", n_tracks=3, random_state=42)
        tra.fit(self.X_train_clf, self.y_train_clf)
        
        # Test predictions
        y_pred = tra.predict(self.X_test_clf)
        self.assertEqual(len(y_pred), len(self.y_test_clf))
        
        # Test probability predictions
        y_proba = tra.predict_proba(self.X_test_clf)
        self.assertEqual(y_proba.shape[0], len(self.y_test_clf))
        self.assertEqual(y_proba.shape[1], len(np.unique(self.y_clf)))
        
        # Test scoring
        score = tra.score(self.X_test_clf, self.y_test_clf)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_regression_basic(self):
        """Test basic regression functionality."""
        tra = OptimizedTRA(task_type="regression", n_tracks=3, random_state=42)
        tra.fit(self.X_train_reg, self.y_train_reg)
        
        # Test predictions
        y_pred = tra.predict(self.X_test_reg)
        self.assertEqual(len(y_pred), len(self.y_test_reg))
        
        # Test scoring (negative MSE)
        score = tra.score(self.X_test_reg, self.y_test_reg)
        self.assertIsInstance(score, float)
        self.assertLessEqual(score, 0.0)  # Negative MSE
    
    def test_invalid_task_type(self):
        """Test invalid task type handling."""
        with self.assertRaises(ValueError):
            tra = OptimizedTRA(task_type="invalid")
            tra.fit(self.X_train_clf, self.y_train_clf)
    
    def test_predict_proba_regression_error(self):
        """Test that predict_proba raises error for regression."""
        tra = OptimizedTRA(task_type="regression", random_state=42)
        tra.fit(self.X_train_reg, self.y_train_reg)
        
        with self.assertRaises(ValueError):
            tra.predict_proba(self.X_test_reg)
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        tra = OptimizedTRA(
            task_type="classification", 
            feature_selection=True, 
            random_state=42
        )
        tra.fit(self.X_train_clf, self.y_train_clf)
        
        # Check that feature selector was created
        self.assertIsNotNone(tra.feature_selector_)
        
        # Test predictions work with feature selection
        y_pred = tra.predict(self.X_test_clf)
        self.assertEqual(len(y_pred), len(self.y_test_clf))
        
    def test_routing_modes_classification(self):
        """Test hard and soft routing modes for classification."""
        for mode in ["hard", "soft"]:
            tra = OptimizedTRA(task_type="classification", n_tracks=3, routing_mode=mode, random_state=42)
            tra.fit(self.X_train_clf, self.y_train_clf)
            
            y_pred = tra.predict(self.X_test_clf)
            self.assertEqual(len(y_pred), len(self.y_test_clf))
            
            y_proba = tra.predict_proba(self.X_test_clf)
            self.assertEqual(y_proba.shape[0], len(self.y_test_clf))
            
    def test_routing_modes_regression(self):
        """Test hard and soft routing modes for regression."""
        for mode in ["hard", "soft"]:
            tra = OptimizedTRA(task_type="regression", n_tracks=3, routing_mode=mode, random_state=42)
            tra.fit(self.X_train_reg, self.y_train_reg)
            
            y_pred = tra.predict(self.X_test_reg)
            self.assertEqual(len(y_pred), len(self.y_test_reg))


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_example_dataset_creation(self):
        """Test example dataset creation."""
        # Classification dataset
        X_clf, y_clf = OptimizedTRA.create_example_dataset(
            task_type="classification", n_samples=100, n_features=5
        )
        self.assertEqual(X_clf.shape, (100, 5))
        self.assertEqual(len(y_clf), 100)
        self.assertGreater(len(np.unique(y_clf)), 1)
        
        # Regression dataset
        X_reg, y_reg = OptimizedTRA.create_example_dataset(
            task_type="regression", n_samples=150, n_features=8
        )
        self.assertEqual(X_reg.shape, (150, 8))
        self.assertEqual(len(y_reg), 150)


if __name__ == '__main__':
    unittest.main()
