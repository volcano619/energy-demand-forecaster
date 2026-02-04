"""
Evaluation Metrics for Time Series Forecasting

Standard forecasting metrics:
- MAPE: Mean Absolute Percentage Error
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error
- SMAPE: Symmetric MAPE
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    
    MAPE = mean(|actual - predicted| / |actual|) * 100
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    More robust when actual values are close to zero.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    
    return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Square Error."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))


def mase(actual: np.ndarray, predicted: np.ndarray, seasonal_period: int = 24) -> float:
    """
    Mean Absolute Scaled Error.
    
    Compares forecast error to naive seasonal forecast.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Naive seasonal forecast error
    naive_errors = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])
    naive_mae = np.mean(naive_errors)
    
    if naive_mae == 0:
        return np.inf
    
    forecast_mae = np.mean(np.abs(actual - predicted))
    return forecast_mae / naive_mae


def coverage(
    actual: np.ndarray, 
    lower: np.ndarray, 
    upper: np.ndarray
) -> float:
    """
    Prediction interval coverage.
    
    Percentage of actual values within [lower, upper] bounds.
    """
    actual = np.array(actual)
    lower = np.array(lower)
    upper = np.array(upper)
    
    within_bounds = (actual >= lower) & (actual <= upper)
    return np.mean(within_bounds) * 100


def evaluate_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray = None,
    upper: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate all metrics for a forecast.
    
    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        'MAPE': mape(actual, predicted),
        'SMAPE': smape(actual, predicted),
        'RMSE': rmse(actual, predicted),
        'MAE': mae(actual, predicted),
        'MASE': mase(actual, predicted)
    }
    
    if lower is not None and upper is not None:
        metrics['Coverage'] = coverage(actual, lower, upper)
    
    return metrics


class ForecastEvaluator:
    """Evaluate and compare multiple forecasting models."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(
        self,
        name: str,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: np.ndarray = None,
        upper: np.ndarray = None
    ) -> Dict[str, float]:
        """Evaluate a single forecast."""
        metrics = evaluate_forecast(actual, predicted, lower, upper)
        self.results[name] = metrics
        return metrics
    
    def compare(self) -> Dict[str, Dict[str, float]]:
        """Compare all evaluated models."""
        return self.results
    
    def best_model(self, metric: str = 'MAPE') -> str:
        """Find best model by specified metric."""
        if not self.results:
            return None
        
        best = min(self.results.items(), key=lambda x: x[1].get(metric, float('inf')))
        return best[0]
    
    def summary(self) -> str:
        """Generate summary report."""
        if not self.results:
            return "No evaluations performed."
        
        lines = ["Forecast Evaluation Summary", "=" * 40]
        
        for name, metrics in self.results.items():
            lines.append(f"\n{name}:")
            for metric, value in metrics.items():
                lines.append(f"  {metric}: {value:.2f}")
        
        best = self.best_model()
        lines.append(f"\nBest Model (by MAPE): {best}")
        
        return "\n".join(lines)
