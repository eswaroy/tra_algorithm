import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import warnings
import joblib
import matplotlib.pyplot as plt
try:
    import networkx as nx
except ImportError:
    nx = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class Record:
    """Simplified Record class for tracking data flow."""
    id: int
    features: np.ndarray
    current_track: str = "track_0"
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)
    confidence: float = 1.0
    
    def update_track(self, new_track: str, confidence: float = 1.0):
        """Update current track with confidence."""
        if new_track != self.current_track:
            self.history.append((self.current_track, time.time()))
            self.current_track = new_track
            self.confidence = confidence
            self.last_update = time.time()

class EnhancedSignalCondition:
    """Enhanced signal condition for track switching with improved regression handling."""
    
    def __init__(self, source_clf, target_clf, threshold: float = 0.1, task_type: str = 'classification'):
        self.source_clf = source_clf
        self.target_clf = target_clf
        self.threshold = threshold
        self.task_type = task_type
        self._variance_cache = None
    
    def _compute_target_variance(self, sample_features: np.ndarray = None):
        """Compute target variance for regression normalization."""
        if self.task_type != 'regression' or self._variance_cache is not None:
            return
        
        # Use a small sample to estimate variance if available
        if sample_features is not None and len(sample_features) > 1:
            try:
                source_preds = self.source_clf.predict(sample_features)
                target_preds = self.target_clf.predict(sample_features)
                combined_preds = np.concatenate([source_preds, target_preds])
                self._variance_cache = np.var(combined_preds) if np.var(combined_preds) > 0 else 1.0
            except:
                self._variance_cache = 1.0
        else:
            self._variance_cache = 1.0
    
    def evaluate(self, record: Record, sample_features: np.ndarray = None) -> Tuple[bool, float]:
        """Evaluate whether to switch tracks with enhanced regression handling."""
        try:
            features = record.features.reshape(1, -1)
            
            if self.task_type == 'classification':
                # Get prediction probabilities
                source_proba = self.source_clf.predict_proba(features)[0]
                target_proba = self.target_clf.predict_proba(features)[0]
                
                # Calculate confidence difference
                source_conf = np.max(source_proba)
                target_conf = np.max(target_proba)
                confidence_diff = target_conf - source_conf
                
                # Switch if target is significantly more confident
                should_switch = confidence_diff > self.threshold
                return should_switch, confidence_diff
                
            else:  # Enhanced regression handling
                # Compute variance for normalization
                self._compute_target_variance(sample_features)
                
                # For regression, use normalized prediction difference
                source_pred = self.source_clf.predict(features)[0]
                target_pred = self.target_clf.predict(features)[0]
                
                # Calculate normalized prediction difference
                pred_diff = abs(target_pred - source_pred)
                normalized_diff = pred_diff / np.sqrt(self._variance_cache) if self._variance_cache > 0 else pred_diff
                
                should_switch = normalized_diff > self.threshold
                return should_switch, normalized_diff
                
        except Exception as e:
            logger.debug(f"Error in EnhancedSignalCondition.evaluate: {str(e)}")
            return False, 0.0

class Signal:
    """Enhanced Signal class for track switching with performance tracking."""
    
    def __init__(self, name: str, condition: EnhancedSignalCondition, source_track: str, target_track: str):
        self.name = name
        self.condition = condition
        self.source_track = source_track
        self.target_track = target_track
        self.activation_count = 0
        self.success_count = 0
        self.confidence = 1.0
        self.last_activation = 0.0
        self.performance_history = deque(maxlen=100)
        
    def evaluate(self, record: Record, sample_features: np.ndarray = None) -> Tuple[bool, float]:
        """Evaluate signal with timing constraints and enhanced condition."""
        current_time = time.time()
        
        # Prevent rapid switching (minimum 0.5 second between switches for better performance)
        if current_time - self.last_activation < 0.5:
            return False, 0.0
        
        # Evaluate condition with sample features for regression variance estimation
        should_switch, switch_confidence = self.condition.evaluate(record, sample_features)
        
        if should_switch:
            self.activation_count += 1
            self.last_activation = current_time
            
        return should_switch, switch_confidence
    
    def update_performance(self, success: bool):
        """Update signal performance metrics with history tracking."""
        self.performance_history.append(success)
        
        if success:
            self.success_count += 1
        
        # Update confidence based on recent performance (last 20 activations)
        if len(self.performance_history) >= 5:
            recent_performance = list(self.performance_history)[-20:]
            self.confidence = sum(recent_performance) / len(recent_performance)
        elif self.activation_count > 0:
            self.confidence = self.success_count / self.activation_count
        else:
            self.confidence = 0.5

class Track:
    """Enhanced Track class with usage tracking and performance metrics."""
    
    def __init__(self, name: str, classifier=None):
        self.name = name
        self.classifier = classifier
        self.signals: List[Signal] = []
        self.records: List[Record] = []
        self.performance_score = 0.5
        self.usage_count = 0
        self.last_used = time.time()
        self.prediction_times = deque(maxlen=50)
        
    def add_signal(self, signal: Signal):
        """Add a signal to this track."""
        self.signals.append(signal)
    
    def process_record(self, record: Record) -> Record:
        """Process a record through this track with usage tracking."""
        self.records.append(record)
        self.usage_count += 1
        self.last_used = time.time()
        return record
    
    def predict(self, X: np.ndarray):
        """Make predictions using this track's classifier with timing."""
        if self.classifier is None:
            raise ValueError(f"No classifier available for track {self.name}")
        
        start_time = time.time()
        result = self.classifier.predict(X)
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        return result
    
    def predict_proba(self, X: np.ndarray):
        """Make probability predictions using this track's classifier with timing."""
        if self.classifier is None:
            raise ValueError(f"No classifier available for track {self.name}")
        
        start_time = time.time()
        result = self.classifier.predict_proba(X)
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        return result
    
    def get_average_prediction_time(self):
        """Get average prediction time for this track."""
        return np.mean(self.prediction_times) if self.prediction_times else 0.0
    
    def is_underused(self, min_usage_threshold: int = 5, time_threshold: float = 300.0):
        """Check if track is underused and candidate for pruning."""
        current_time = time.time()
        return (self.usage_count < min_usage_threshold and 
                current_time - self.last_used > time_threshold)

class OptimizedTRA(BaseEstimator, ClassifierMixin, RegressorMixin):
    """Optimized Track/Rail Algorithm with enhanced performance and parallel processing."""
    
    def __init__(self, 
                 task_type: str = "classification",
                 n_tracks: int = 3,
                 signal_threshold: float = 0.1,
                 random_state: Optional[int] = None,
                 n_estimators: int = 50,
                 max_depth: int = 6,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 4,
                 feature_selection: bool = True,
                 handle_imbalanced: bool = True,
                 parallel_signals: bool = True,
                 max_workers: int = 4,
                 enable_track_pruning: bool = True,
                 pruning_interval: int = 100):
        
        # Validate task_type
        if task_type not in ("classification", "regression"):
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'.")
        
        self.task_type = task_type
        self.n_tracks = n_tracks
        self.signal_threshold = signal_threshold
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_selection = feature_selection
        self.handle_imbalanced = handle_imbalanced
        self.parallel_signals = parallel_signals
        self.max_workers = min(max_workers, 8)  # Limit max workers
        self.enable_track_pruning = enable_track_pruning
        self.pruning_interval = pruning_interval
        
        # Initialize components
        self.tracks: Dict[str, Track] = {}
        self.scaler_ = StandardScaler()
        self.feature_selector_ = None
        self.fitted_ = False
        self.classes_ = None
        self.n_features_in_ = None
        self.class_weights_ = None
        self.prediction_count_ = 0
        self.sample_features_cache_ = None
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_base_estimator(self):
        """Create base estimator with optimized parameters."""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'n_jobs': 1  # Keep at 1 for better control over parallelization
        }
        
        if self.task_type == "classification":
            if self.class_weights_ is not None:
                params['class_weight'] = 'balanced'
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)
    
    def _setup_feature_selection(self, X: np.ndarray, y: np.ndarray):
        """Setup feature selection with enhanced parameters."""
        if not self.feature_selection:
            return
            
        # More aggressive feature selection for better performance
        n_features = min(X.shape[1], max(3, X.shape[1] // 3))
        
        if self.task_type == "classification":
            score_func = f_classif
        else:
            score_func = f_regression
            
        self.feature_selector_ = SelectKBest(score_func=score_func, k=n_features)
        self.feature_selector_.fit(X, y)
        logger.info(f"Selected {n_features} features out of {X.shape[1]}")
    
    def _handle_class_imbalance(self, y: np.ndarray):
        """Handle class imbalance by computing class weights."""
        if self.task_type == 'classification' and self.handle_imbalanced:
            try:
                classes = np.unique(y)
                class_weights = compute_class_weight('balanced', classes=classes, y=y)
                self.class_weights_ = dict(zip(classes, class_weights))
                logger.info(f"Computed class weights: {self.class_weights_}")
            except Exception as e:
                logger.warning(f"Class weight computation failed: {str(e)}")
                self.class_weights_ = None
    
    def _create_tracks(self, X: np.ndarray, y: np.ndarray):
        """Create and train tracks with enhanced sampling."""
        logger.info(f"Creating {self.n_tracks} tracks...")
        
        n_samples = X.shape[0]
        
        # Cache sample features for regression variance estimation
        if self.task_type == 'regression':
            sample_size = min(100, n_samples // 4)
            sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
            self.sample_features_cache_ = X[sample_indices]
        
        for i in range(self.n_tracks):
            track_name = f"track_{i}"
            
            # Create stratified bootstrap sample for better diversity
            if self.task_type == 'classification' and len(np.unique(y)) > 1:
                try:
                    from sklearn.utils import resample
                    X_track, y_track = resample(X, y, n_samples=n_samples, 
                                              random_state=self.random_state + i,
                                              stratify=y)
                except:
                    # Fallback to regular bootstrap
                    indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    X_track = X[indices]
                    y_track = y[indices]
            else:
                # Regular bootstrap for regression or single class
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_track = X[indices]
                y_track = y[indices]
            
            # Train classifier for this track
            clf = self._get_base_estimator()
            clf.fit(X_track, y_track)
            
            # Create track
            track = Track(track_name, clf)
            self.tracks[track_name] = track
            
        logger.info(f"Created {len(self.tracks)} tracks")
    
    def _create_signals(self):
        """Create enhanced signals between tracks."""
        logger.info("Creating enhanced signals between tracks...")
        
        track_names = list(self.tracks.keys())
        signal_count = 0
        
        for i, source_track in enumerate(track_names):
            for j, target_track in enumerate(track_names):
                if i != j:
                    signal_name = f"signal_{source_track}_to_{target_track}"
                    
                    # Adjust threshold based on task type
                    adjusted_threshold = self.signal_threshold
                    if self.task_type == 'regression':
                        adjusted_threshold = self.signal_threshold * 0.5  # More sensitive for regression
                    
                    condition = EnhancedSignalCondition(
                        source_clf=self.tracks[source_track].classifier,
                        target_clf=self.tracks[target_track].classifier,
                        threshold=adjusted_threshold,
                        task_type=self.task_type
                    )
                    
                    signal = Signal(
                        name=signal_name,
                        condition=condition,
                        source_track=source_track,
                        target_track=target_track
                    )
                    
                    self.tracks[source_track].add_signal(signal)
                    signal_count += 1
        
        logger.info(f"Created {signal_count} enhanced signals")
    
    def _evaluate_signals_parallel(self, record: Record) -> str:
        """Evaluate signals in parallel for better performance."""
        current_track = record.current_track
        
        if current_track not in self.tracks:
            return "track_0"
        
        track = self.tracks[current_track]
        if not track.signals:
            return current_track
        
        if not self.parallel_signals or len(track.signals) < 3:
            # Use sequential evaluation for small numbers of signals
            return self._evaluate_signals_sequential(record)
        
        best_signal = None
        best_confidence = 0.0
        
        def evaluate_signal(signal):
            should_switch, confidence = signal.evaluate(record, self.sample_features_cache_)
            return signal, should_switch, confidence
        
        try:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(track.signals))) as executor:
                future_to_signal = {
                    executor.submit(evaluate_signal, signal): signal 
                    for signal in track.signals
                }
                
                for future in as_completed(future_to_signal, timeout=1.0):
                    try:
                        signal, should_switch, confidence = future.result()
                        if should_switch and confidence > best_confidence:
                            best_signal = signal
                            best_confidence = confidence
                    except Exception as e:
                        logger.debug(f"Signal evaluation error: {str(e)}")
                        continue
        except Exception as e:
            logger.debug(f"Parallel evaluation failed, falling back to sequential: {str(e)}")
            return self._evaluate_signals_sequential(record)
        
        # Switch to best target track if found
        if best_signal:
            return best_signal.target_track
        else:
            return current_track
    
    def _evaluate_signals_sequential(self, record: Record) -> str:
        """Sequential signal evaluation fallback."""
        current_track = record.current_track
        
        if current_track not in self.tracks:
            return "track_0"
        
        track = self.tracks[current_track]
        best_signal = None
        best_confidence = 0.0
        
        # Evaluate all signals from current track
        for signal in track.signals:
            should_switch, confidence = signal.evaluate(record, self.sample_features_cache_)
            if should_switch and confidence > best_confidence:
                best_signal = signal
                best_confidence = confidence
        
        # Switch to best target track if found
        if best_signal:
            return best_signal.target_track
        else:
            return current_track
    
    def _prune_unused_tracks(self):
        """Remove underused tracks to optimize memory and computation."""
        if not self.enable_track_pruning or len(self.tracks) <= 2:
            return
        
        tracks_to_remove = []
        for track_name, track in self.tracks.items():
            if track.is_underused() and track_name != "track_0":  # Always keep track_0
                tracks_to_remove.append(track_name)
        
        if tracks_to_remove and len(self.tracks) - len(tracks_to_remove) >= 2:
            for track_name in tracks_to_remove:
                # Remove signals pointing to this track
                for remaining_track in self.tracks.values():
                    remaining_track.signals = [
                        s for s in remaining_track.signals 
                        if s.target_track != track_name
                    ]
                
                # Remove the track
                del self.tracks[track_name]
                logger.info(f"Pruned unused track: {track_name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the enhanced TRA model."""
        start_time = time.time()
        logger.info("Fitting Enhanced Optimized TRA model...")
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        
        # Store classes for classification
        if self.task_type == "classification":
            self.classes_ = np.unique(y)
            logger.info(f"Found {len(self.classes_)} classes: {self.classes_}")
        
        # Handle class imbalance
        self._handle_class_imbalance(y)
        
        # Scale features
        X_scaled = self.scaler_.fit_transform(X)
        
        # Enhanced feature selection
        self._setup_feature_selection(X_scaled, y)
        if self.feature_selector_ is not None:
            X_selected = self.feature_selector_.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # Create tracks and enhanced signals
        self._create_tracks(X_selected, y)
        self._create_signals()
        
        # Validate model performance with enhanced cross-validation
        try:
            sample_track = self.tracks["track_0"]
            cv_scores = cross_val_score(sample_track.classifier, X_selected, y, cv=5, scoring=None)
            logger.info(f"Cross-validation scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Update track performance scores
            for i, (track_name, track) in enumerate(self.tracks.items()):
                if i < len(cv_scores):
                    track.performance_score = cv_scores[i]
                    
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
        
        training_time = time.time() - start_time
        logger.info(f"Enhanced training completed in {training_time:.2f}s")
        
        self.fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the enhanced TRA model with optimizations."""
        check_is_fitted(self, 'fitted_')
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but TRA was fitted with {self.n_features_in_} features")
        
        # Transform features
        X_scaled = self.scaler_.transform(X)
        if self.feature_selector_ is not None:
            X_selected = self.feature_selector_.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        predictions = []
        
        for i, features in enumerate(X_selected):
            # Create record
            record = Record(id=i, features=features, current_track="track_0")
            
            # Evaluate signals to find best track with optimized iterations
            max_iterations = 3  # Reduced for better performance
            for iteration in range(max_iterations):
                new_track = self._evaluate_signals_parallel(record)
                if new_track == record.current_track:
                    break
                record.update_track(new_track)
            
            # Make prediction using final track
            if record.current_track in self.tracks:
                track = self.tracks[record.current_track]
                track.process_record(record)  # Track usage
                pred = track.predict(features.reshape(1, -1))[0]
            else:
                # Fallback to first available track
                available_tracks = list(self.tracks.keys())
                if available_tracks:
                    track = self.tracks[available_tracks[0]]
                    pred = track.predict(features.reshape(1, -1))[0]
                else:
                    raise ValueError("No tracks available for prediction")
            
            predictions.append(pred)
            
            # Periodic track pruning
            self.prediction_count_ += 1
            if (self.enable_track_pruning and 
                self.prediction_count_ % self.pruning_interval == 0):
                self._prune_unused_tracks()
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities with optimizations (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        check_is_fitted(self, 'fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Transform features
        X_scaled = self.scaler_.transform(X)
        if self.feature_selector_ is not None:
            X_selected = self.feature_selector_.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        probabilities = []
        
        for i, features in enumerate(X_selected):
            # Create record
            record = Record(id=i, features=features, current_track="track_0")
            
            # Evaluate signals to find best track
            max_iterations = 3
            for iteration in range(max_iterations):
                new_track = self._evaluate_signals_parallel(record)
                if new_track == record.current_track:
                    break
                record.update_track(new_track)
            
            # Make prediction using final track
            if record.current_track in self.tracks:
                track = self.tracks[record.current_track]
                track.process_record(record)
                proba = track.predict_proba(features.reshape(1, -1))[0]
            else:
                # Fallback to first available track
                available_tracks = list(self.tracks.keys())
                if available_tracks:
                    track = self.tracks[available_tracks[0]]
                    proba = track.predict_proba(features.reshape(1, -1))[0]
                else:
                    raise ValueError("No tracks available for prediction")
            
            probabilities.append(proba)
        
        return np.array(probabilities)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate enhanced model score."""
        y_pred = self.predict(X)
        if self.task_type == "classification":
            return accuracy_score(y, y_pred)
        else:
            return -mean_squared_error(y, y_pred)
    
    def visualize(self, output_file: str = None, figsize: Tuple[int, int] = (14, 10)):
        """Visualize the enhanced TRA structure with performance metrics."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before visualization")
        
        if nx is None:
            logger.warning("NetworkX not available, skipping visualization")
            return
        
        G = nx.DiGraph()
        
        # Add nodes (tracks) with performance information
        for track_name, track in self.tracks.items():
            avg_time = track.get_average_prediction_time()
            G.add_node(track_name, 
                      performance=track.performance_score,
                      usage=track.usage_count,
                      avg_time=avg_time)
        
        # Add edges (signals) with confidence weights
        for track in self.tracks.values():
            for signal in track.signals:
                G.add_edge(
                    signal.source_track,
                    signal.target_track,
                    weight=signal.confidence,
                    activations=signal.activation_count,
                    label=f"{signal.name}\nConf: {signal.confidence:.2f}\nAct: {signal.activation_count}"
                )
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, k=4, iterations=100)
        
        # Color nodes by performance
        node_colors = [self.tracks[node].performance_score for node in G.nodes()]
        # Draw nodes with size based on usage
        node_sizes = [max(1000, self.tracks[node].usage_count * 50) for node in G.nodes()]
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=node_sizes, alpha=0.8, cmap='RdYlBu')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        # Draw edges with width based on confidence
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, 
                             width=edge_weights, alpha=0.6, edge_color='gray')
        
        plt.title("Enhanced Optimized TRA Structure\n(Node color: performance, Node size: usage, Edge width: confidence)", 
                 fontsize=14, fontweight='bold')
        plt.colorbar(nodes, label='Track Performance')
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Enhanced visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics about tracks and signals."""
        if not self.fitted_:
            return {}
        
        stats = {
            'n_tracks': len(self.tracks),
            'n_signals': sum(len(track.signals) for track in self.tracks.values()),
            'total_predictions': self.prediction_count_,
            'parallel_processing': self.parallel_signals,
            'track_pruning_enabled': self.enable_track_pruning,
            'track_details': {}
        }
        
        total_usage = sum(track.usage_count for track in self.tracks.values())
        
        for track_name, track in self.tracks.items():
            usage_pct = (track.usage_count / total_usage * 100) if total_usage > 0 else 0
            stats['track_details'][track_name] = {
                'n_records': len(track.records),
                'n_signals': len(track.signals),
                'usage_count': track.usage_count,
                'usage_percentage': usage_pct,
                'avg_signal_confidence': np.mean([s.confidence for s in track.signals]) if track.signals else 0.0,
                'performance_score': track.performance_score,
                'avg_prediction_time': track.get_average_prediction_time(),
                'last_used': track.last_used,
                'is_underused': track.is_underused()
            }
        
        return stats
    
    def save_model(self, filename: str):
        """Save the trained model with enhanced metadata."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before saving")
        
        # Create save data with metadata
        save_data = {
            'model': self,
            'metadata': {
                'task_type': self.task_type,
                'n_tracks': len(self.tracks),
                'n_features': self.n_features_in_,
                'parallel_enabled': self.parallel_signals,
                'pruning_enabled': self.enable_track_pruning,
                'prediction_count': self.prediction_count_,
                'save_timestamp': time.time()
            }
        }
        
        joblib.dump(save_data, filename)
        logger.info(f"Enhanced model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename: str) -> 'OptimizedTRA':
        """Load a trained model with metadata validation."""
        save_data = joblib.load(filename)
        
        if isinstance(save_data, dict) and 'model' in save_data:
            model = save_data['model']
            metadata = save_data.get('metadata', {})
            logger.info(f"Enhanced model loaded from {filename}")
            logger.info(f"Model metadata: {metadata}")
        else:
            # Backward compatibility
            model = save_data
            logger.info(f"Legacy model loaded from {filename}")
        
        return model
    
    def get_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.fitted_:
            return "Model not fitted yet."
        
        stats = self.get_track_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("ENHANCED OPTIMIZED TRA PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Task Type: {self.task_type}")
        report.append(f"Number of Tracks: {stats['n_tracks']}")
        report.append(f"Number of Signals: {stats['n_signals']}")
        report.append(f"Total Predictions Made: {stats['total_predictions']}")
        report.append(f"Parallel Processing: {'Enabled' if stats['parallel_processing'] else 'Disabled'}")
        report.append(f"Track Pruning: {'Enabled' if stats['track_pruning_enabled'] else 'Disabled'}")
        report.append("")
        
        report.append("TRACK PERFORMANCE DETAILS:")
        report.append("-" * 40)
        
        for track_name, details in stats['track_details'].items():
            report.append(f"Track: {track_name}")
            report.append(f"  Usage Count: {details['usage_count']} ({details['usage_percentage']:.1f}%)")
            report.append(f"  Performance Score: {details['performance_score']:.4f}")
            report.append(f"  Avg Prediction Time: {details['avg_prediction_time']:.6f}s")
            report.append(f"  Number of Signals: {details['n_signals']}")
            report.append(f"  Avg Signal Confidence: {details['avg_signal_confidence']:.4f}")
            report.append(f"  Underused: {'Yes' if details['is_underused'] else 'No'}")
            report.append("")
        
        # Signal performance summary
        all_signals = []
        for track in self.tracks.values():
            all_signals.extend(track.signals)
        
        if all_signals:
            report.append("SIGNAL PERFORMANCE SUMMARY:")
            report.append("-" * 40)
            total_activations = sum(s.activation_count for s in all_signals)
            avg_confidence = np.mean([s.confidence for s in all_signals])
            active_signals = sum(1 for s in all_signals if s.activation_count > 0)
            
            report.append(f"Total Signal Activations: {total_activations}")
            report.append(f"Average Signal Confidence: {avg_confidence:.4f}")
            report.append(f"Active Signals: {active_signals}/{len(all_signals)}")
            
            # Top performing signals
            top_signals = sorted(all_signals, key=lambda x: x.confidence, reverse=True)[:3]
            report.append("\nTop 3 Performing Signals:")
            for i, signal in enumerate(top_signals, 1):
                report.append(f"  {i}. {signal.name}: {signal.confidence:.4f} confidence, "
                            f"{signal.activation_count} activations")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def optimize_parameters(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Optimize signal thresholds based on validation performance."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before parameter optimization")
        
        logger.info("Optimizing signal parameters...")
        
        original_threshold = self.signal_threshold
        best_threshold = original_threshold
        best_score = self.score(X_val, y_val)
        
        # Test different thresholds
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        for threshold in thresholds:
            if threshold == original_threshold:
                continue
                
            # Update signal thresholds
            for track in self.tracks.values():
                for signal in track.signals:
                    signal.condition.threshold = threshold
            
            # Evaluate performance
            try:
                score = self.score(X_val, y_val)
                logger.info(f"Threshold {threshold}: Score {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            except Exception as e:
                logger.warning(f"Error evaluating threshold {threshold}: {str(e)}")
        
        # Apply best threshold
        self.signal_threshold = best_threshold
        for track in self.tracks.values():
            for signal in track.signals:
                signal.condition.threshold = best_threshold
        
        optimization_results = {
            'original_threshold': original_threshold,
            'optimized_threshold': best_threshold,
            'original_score': self.score(X_val, y_val) if best_threshold != original_threshold else best_score,
            'optimized_score': best_score,
            'improvement': best_score - (self.score(X_val, y_val) if best_threshold != original_threshold else best_score)
        }
        
        logger.info(f"Parameter optimization completed. Best threshold: {best_threshold}")
        return optimization_results
    
    @staticmethod
    def create_example_dataset(task_type: str = "classification", 
                             n_samples: int = 1000, 
                             n_features: int = 10,
                             random_state: int = 42,
                             noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Create enhanced example dataset for testing with more realistic characteristics."""
        np.random.seed(random_state)
        
        if task_type == "classification":
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(3, n_features // 2),
                n_redundant=max(1, n_features // 4),
                n_classes=3,
                n_clusters_per_class=2,  # More complex clusters
                flip_y=noise_level,  # Add label noise
                random_state=random_state,
                class_sep=0.8  # Moderate class separation
            )
        else:
            from sklearn.datasets import make_regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(3, n_features // 2),
                noise=noise_level * 10,  # Scaled noise for regression
                random_state=random_state,
                bias=10.0  # Add bias term
            )
        
        return X, y

def run_enhanced_example():
    """Run enhanced demonstration of Optimized TRA with all improvements."""
    logger.info("=" * 70)
    logger.info("ENHANCED OPTIMIZED TRACK/RAIL ALGORITHM (TRA) DEMONSTRATION")
    logger.info("=" * 70)
    
    for task_type in ["classification", "regression"]:
        logger.info(f"\n{task_type.upper()} EXAMPLE WITH OPTIMIZATIONS")
        logger.info("-" * 50)
        
        # Create enhanced dataset
        X, y = OptimizedTRA.create_example_dataset(
            task_type=task_type, 
            n_samples=1200,
            n_features=15,
            noise_level=0.15
        )
        logger.info(f"Enhanced dataset created: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data with validation set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Create and train enhanced model
        tra = OptimizedTRA(
            task_type=task_type,
            n_tracks=5,  # More tracks for better performance
            random_state=42,
            n_estimators=40,
            max_depth=6,
            feature_selection=True,
            handle_imbalanced=True,
            parallel_signals=True,
            max_workers=4,
            enable_track_pruning=True,
            pruning_interval=50,
            signal_threshold=0.15
        )
        
        # Train model
        start_time = time.time()
        tra.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Optimize parameters using validation set
        optimization_results = tra.optimize_parameters(X_val, y_val)
        
        # Make predictions
        start_time = time.time()
        y_pred = tra.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Evaluate performance
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info(f"Test F1-score: {f1:.4f}")
            
            # Test probability predictions
            try:
                y_proba = tra.predict_proba(X_test)
                logger.info(f"Probability predictions shape: {y_proba.shape}")
            except Exception as e:
                logger.warning(f"Probability prediction failed: {str(e)}")
                
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            logger.info(f"Test MSE: {mse:.4f}")
            logger.info(f"Test RMSE: {rmse:.4f}")
        
        logger.info(f"Training time: {training_time:.2f}s")
        logger.info(f"Prediction time: {prediction_time:.4f}s ({prediction_time/len(X_test)*1000:.2f}ms per sample)")
        
        # Parameter optimization results
        logger.info(f"Parameter Optimization Results:")
        logger.info(f"  Original threshold: {optimization_results['original_threshold']}")
        logger.info(f"  Optimized threshold: {optimization_results['optimized_threshold']}")
        logger.info(f"  Performance improvement: {optimization_results['improvement']:.4f}")
        
        # Generate performance report
        performance_report = tra.get_performance_report()
        logger.info("\n" + performance_report)
        
        # Test enhanced visualization
        try:
            tra.visualize(f"enhanced_tra_{task_type}_structure.png")
            logger.info("Enhanced visualization created successfully")
        except Exception as e:
            logger.warning(f"Visualization failed: {str(e)}")
        
        # Test enhanced model saving/loading
        model_filename = f"enhanced_tra_{task_type}_model.joblib"
        try:
            tra.save_model(model_filename)
            loaded_tra = OptimizedTRA.load_model(model_filename)
            loaded_pred = loaded_tra.predict(X_test[:10])
            logger.info(f"Enhanced model save/load test successful: {len(loaded_pred)} predictions")
        except Exception as e:
            logger.warning(f"Model save/load failed: {str(e)}")
        
        # Test scoring with timing
        try:
            start_time = time.time()
            score = tra.score(X_test, y_test)
            scoring_time = time.time() - start_time
            logger.info(f"Model score: {score:.4f} (computed in {scoring_time:.4f}s)")
        except Exception as e:
            logger.warning(f"Scoring failed: {str(e)}")

if __name__ == "__main__":
    try:
        run_enhanced_example()
        logger.info("\n" + "=" * 70)
        logger.info("ENHANCED TRA DEMONSTRATION COMPLETED SUCCESSFULLY!")
        logger.info("Key Improvements:")
        logger.info("✓ Enhanced signal conditions with regression optimization")
        logger.info("✓ Parallel signal evaluation for better performance")
        logger.info("✓ Automatic track pruning for memory optimization")
        logger.info("✓ Advanced performance monitoring and reporting")
        logger.info("✓ Parameter optimization with validation data")
        logger.info("✓ Improved visualization with performance metrics")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Error during enhanced demonstration: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
