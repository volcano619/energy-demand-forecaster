"""
Ensemble Model for Energy Demand Forecasting

Combines predictions from multiple models with weighted averaging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from config import ENSEMBLE_WEIGHTS, DATETIME_COL, TARGET_COL

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Ensemble forecaster that combines multiple models.
    
    Supports:
    - Weighted averaging
    - Automatic weight optimization
    - Confidence intervals
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None
    ):
        self.weights = weights or ENSEMBLE_WEIGHTS.copy()
        self.models = {}
        self.is_fitted = False
    
    def add_model(self, name: str, model: object, weight: float = None):
        """Add a model to the ensemble."""
        self.models[name] = model
        if weight is not None:
            self.weights[name] = weight
        
        # Normalize weights
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure weights sum to 1."""
        total = sum(self.weights.get(name, 0) for name in self.models.keys())
        if total > 0:
            for name in self.models.keys():
                if name in self.weights:
                    self.weights[name] /= total
    
    def fit(
        self,
        df: pd.DataFrame,
        X: np.ndarray = None,
        y: np.ndarray = None,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> 'EnsembleForecaster':
        """
        Fit all models in the ensemble.
        
        Args:
            df: DataFrame for Prophet-style models
            X, y: Arrays for LSTM-style models
            X_val, y_val: Validation data for weight optimization
        """
        for name, model in self.models.items():
            logger.info(f"Fitting {name}...")
            
            if hasattr(model, 'fit'):
                # Check model type
                if 'Prophet' in type(model).__name__ or 'Seasonal' in type(model).__name__:
                    model.fit(df)
                elif X is not None and y is not None:
                    model.fit(X, y)
        
        self.is_fitted = True
        
        # Optimize weights if validation data provided
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)
        
        return self
    
    def _optimize_weights(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        metric: str = 'mape'
    ):
        """Optimize weights based on validation performance."""
        errors = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(len(y_val))
                    if len(pred) == len(y_val):
                        mape = np.mean(np.abs((y_val - pred) / (y_val + 1e-8))) * 100
                        errors[name] = mape
            except Exception as e:
                logger.warning(f"Could not evaluate {name}: {e}")
        
        if errors:
            # Inverse error weighting (lower error = higher weight)
            total_inv_error = sum(1.0 / (e + 1e-8) for e in errors.values())
            for name in errors:
                self.weights[name] = (1.0 / (errors[name] + 1e-8)) / total_inv_error
            
            logger.info(f"Optimized weights: {self.weights}")
    
    def predict(
        self,
        periods: int,
        return_individual: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate ensemble predictions.
        
        Returns:
            Dictionary with 'ensemble', 'lower', 'upper' predictions
            Optionally includes individual model predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        predictions = {}
        all_preds = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(periods)
                
                # Handle Prophet-style output
                if isinstance(pred, pd.DataFrame):
                    pred_values = pred['yhat'].values
                else:
                    pred_values = pred
                
                predictions[name] = pred_values
                all_preds.append(pred_values * self.weights.get(name, 0))
                
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
        
        if not all_preds:
            raise ValueError("No predictions generated")
        
        # Weighted ensemble
        ensemble_pred = np.sum(all_preds, axis=0)
        predictions['ensemble'] = ensemble_pred
        
        # Confidence intervals (based on model disagreement)
        if len(all_preds) > 1:
            std = np.std([p / max(self.weights.values()) for p in all_preds], axis=0)
            predictions['lower'] = ensemble_pred - 1.96 * std
            predictions['upper'] = ensemble_pred + 1.96 * std
        else:
            predictions['lower'] = ensemble_pred * 0.9
            predictions['upper'] = ensemble_pred * 1.1
        
        if not return_individual:
            return {
                'ensemble': predictions['ensemble'],
                'lower': predictions['lower'],
                'upper': predictions['upper']
            }
        
        return predictions
    
    def get_model_contributions(self) -> Dict[str, float]:
        """Get contribution percentage of each model."""
        return {name: weight * 100 for name, weight in self.weights.items() if name in self.models}
