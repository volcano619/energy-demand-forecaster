"""
Statistical Models for Energy Demand Forecasting

Implements:
1. Prophet - Facebook's time series forecasting
2. SARIMA - Seasonal ARIMA (optional, simpler fallback)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

from config import (
    TARGET_COL, DATETIME_COL,
    PROPHET_YEARLY_SEASONALITY, PROPHET_WEEKLY_SEASONALITY,
    PROPHET_DAILY_SEASONALITY, PROPHET_CHANGEPOINT_PRIOR_SCALE
)

logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Facebook Prophet for time series forecasting.
    
    Strengths:
    - Handles multiple seasonalities well
    - Robust to missing data
    - Incorporates holidays
    - Provides uncertainty intervals
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = PROPHET_YEARLY_SEASONALITY,
        weekly_seasonality: bool = PROPHET_WEEKLY_SEASONALITY,
        daily_seasonality: bool = PROPHET_DAILY_SEASONALITY,
        changepoint_prior_scale: float = PROPHET_CHANGEPOINT_PRIOR_SCALE
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self.is_fitted = False
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to Prophet format (ds, y)."""
        prophet_df = df[[DATETIME_COL, TARGET_COL]].copy()
        prophet_df.columns = ['ds', 'y']
        return prophet_df
    
    def fit(
        self, 
        df: pd.DataFrame,
        regressors: List[str] = None
    ) -> 'ProphetForecaster':
        """
        Fit Prophet model on training data.
        
        Args:
            df: DataFrame with timestamp and target columns
            regressors: Optional list of additional regressor columns
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.error("Prophet not installed. Install with: pip install prophet")
            raise
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        # Add regressors if provided
        if regressors:
            for reg in regressors:
                self.model.add_regressor(reg)
        
        prophet_df = self._prepare_data(df)
        
        # Add regressors to dataframe
        if regressors:
            for reg in regressors:
                prophet_df[reg] = df[reg].values
        
        logger.info(f"Fitting Prophet on {len(prophet_df)} samples...")
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        return self
    
    def predict(
        self, 
        periods: int,
        include_history: bool = False,
        future_regressors: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            periods: Number of periods to forecast
            include_history: Include historical predictions
            future_regressors: DataFrame with future regressor values
            
        Returns:
            DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='H')
        
        # Add regressors if needed
        if future_regressors is not None:
            for col in future_regressors.columns:
                future[col] = future_regressors[col].values[:len(future)]
        
        forecast = self.model.predict(future)
        
        if not include_history:
            forecast = forecast.tail(periods)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def get_components(self, forecast: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """Extract trend, seasonality components."""
        if forecast is None and not self.is_fitted:
            raise ValueError("No forecast available")
        
        components = {}
        
        if hasattr(self.model, 'history'):
            future = self.model.make_future_dataframe(periods=0, freq='H')
            full_forecast = self.model.predict(future)
            
            components['trend'] = full_forecast['trend']
            
            if self.yearly_seasonality:
                components['yearly'] = full_forecast.get('yearly', pd.Series())
            if self.weekly_seasonality:
                components['weekly'] = full_forecast.get('weekly', pd.Series())
            if self.daily_seasonality:
                components['daily'] = full_forecast.get('daily', pd.Series())
        
        return components


class SimpleSeasonalModel:
    """
    Simple seasonal naive model as baseline.
    
    Predicts using same value from previous period:
    - Same hour yesterday
    - Same hour last week
    """
    
    def __init__(self, seasonal_period: int = 168):
        """
        Args:
            seasonal_period: Hours in seasonal cycle (168 = 1 week)
        """
        self.seasonal_period = seasonal_period
        self.history = None
    
    def fit(self, df: pd.DataFrame) -> 'SimpleSeasonalModel':
        """Store historical data for naive forecasting."""
        self.history = df[[DATETIME_COL, TARGET_COL]].copy()
        self.history.set_index(DATETIME_COL, inplace=True)
        return self
    
    def predict(self, periods: int) -> np.ndarray:
        """Predict by repeating seasonal pattern."""
        if self.history is None:
            raise ValueError("Model not fitted")
        
        # Use last seasonal_period values and repeat
        last_values = self.history[TARGET_COL].values[-self.seasonal_period:]
        
        predictions = []
        for i in range(periods):
            predictions.append(last_values[i % self.seasonal_period])
        
        return np.array(predictions)


class MovingAverageModel:
    """Simple moving average for comparison."""
    
    def __init__(self, window: int = 24):
        self.window = window
        self.last_values = None
    
    def fit(self, df: pd.DataFrame) -> 'MovingAverageModel':
        self.last_values = df[TARGET_COL].values[-self.window:]
        return self
    
    def predict(self, periods: int) -> np.ndarray:
        predictions = []
        values = list(self.last_values)
        
        for _ in range(periods):
            pred = np.mean(values[-self.window:])
            predictions.append(pred)
            values.append(pred)
        
        return np.array(predictions)
