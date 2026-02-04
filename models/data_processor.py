"""
Data Processor for Energy Demand Forecasting

Handles:
1. Synthetic data generation
2. Feature engineering
3. Train/test splitting
4. Data normalization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import logging

from config import (
    DATA_START_DATE, DATA_END_DATE, FREQUENCY,
    TARGET_COL, DATETIME_COL, TRAIN_RATIO, VALIDATION_RATIO,
    LSTM_SEQUENCE_LENGTH, DATA_FILE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_energy_data(
    start_date: str = DATA_START_DATE,
    end_date: str = DATA_END_DATE,
    save_path: str = None
) -> pd.DataFrame:
    """
    Generate realistic synthetic energy consumption data.
    
    Patterns incorporated:
    - Daily seasonality (peak at noon/evening, low at night)
    - Weekly seasonality (lower on weekends)
    - Yearly seasonality (higher in summer/winter for AC/heating)
    - Temperature correlation
    - Random noise
    - Occasional anomalies
    """
    np.random.seed(42)
    
    # Generate hourly timestamps
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    n_samples = len(date_range)
    
    logger.info(f"Generating {n_samples} hourly samples from {start_date} to {end_date}")
    
    # Base load (MW)
    base_load = 500
    
    # Extract time features
    hours = date_range.hour
    days = date_range.dayofweek
    months = date_range.month
    
    # Generate temperature (correlated with demand)
    # Seasonal temperature pattern
    temp_seasonal = 15 + 15 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
    # Daily temperature variation
    temp_daily = 5 * np.sin(2 * np.pi * (hours - 6) / 24)
    # Random noise
    temp_noise = np.random.normal(0, 3, n_samples)
    temperature = temp_seasonal + temp_daily + temp_noise
    
    # Humidity (inversely correlated with temp somewhat)
    humidity = 60 - 0.3 * temperature + np.random.normal(0, 10, n_samples)
    humidity = np.clip(humidity, 20, 95)
    
    # Daily pattern (MW variation)
    # Peak at 9-11am and 6-9pm, low at 3-5am
    daily_pattern = (
        100 * np.sin(np.pi * (hours - 6) / 12) * (hours >= 6) * (hours <= 18) +
        80 * np.sin(np.pi * (hours - 18) / 6) * (hours >= 18) +
        -50 * (hours >= 0) * (hours <= 5)
    )
    
    # Weekly pattern (lower on weekends)
    weekly_pattern = np.where(days >= 5, -80, 0)  # Sat=5, Sun=6
    
    # Seasonal pattern (higher in summer and winter)
    # Peak in July/August (AC) and December/January (heating)
    seasonal_pattern = 100 * np.cos(2 * np.pi * (months - 1) / 6)
    
    # Temperature effect on demand
    # Demand increases when temp deviates from comfortable range (18-22°C)
    temp_effect = 3 * np.abs(temperature - 20)
    
    # Combine all patterns
    demand = (
        base_load +
        daily_pattern +
        weekly_pattern +
        seasonal_pattern +
        temp_effect +
        np.random.normal(0, 20, n_samples)  # Random noise
    )
    
    # Convert to numpy array explicitly
    demand = np.array(demand)
    
    # Add occasional anomalies (2% of data)
    anomaly_mask = np.random.random(n_samples) < 0.02
    anomaly_multipliers = np.random.choice([0.7, 1.4], size=n_samples)
    demand = np.where(anomaly_mask, demand * anomaly_multipliers, demand)
    
    # Ensure positive demand
    demand = np.maximum(demand, 100)
    
    # US Federal Holidays (simplified)
    us_holidays = [
        "2022-01-01", "2022-01-17", "2022-02-21", "2022-05-30", "2022-07-04",
        "2022-09-05", "2022-10-10", "2022-11-11", "2022-11-24", "2022-12-25",
        "2023-01-01", "2023-01-16", "2023-02-20", "2023-05-29", "2023-07-04",
        "2023-09-04", "2023-10-09", "2023-11-10", "2023-11-23", "2023-12-25"
    ]
    holiday_dates = set(pd.to_datetime(us_holidays).date)
    is_holiday = pd.Series(date_range.date).isin(holiday_dates).astype(int).values
    
    # Create DataFrame
    df = pd.DataFrame({
        DATETIME_COL: date_range,
        TARGET_COL: demand.round(2),
        'temperature': temperature.round(1),
        'humidity': humidity.round(1),
        'hour': hours,
        'day_of_week': days,
        'month': months,
        'is_weekend': (days >= 5).astype(int),
        'is_holiday': is_holiday,
        'is_anomaly': anomaly_mask.astype(int)
    })
    
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved data to {save_path}")
    
    return df


def load_energy_data(file_path: str = None) -> pd.DataFrame:
    """Load energy data from CSV."""
    path = file_path or DATA_FILE
    
    if not path.exists():
        logger.info("Data file not found, generating synthetic data...")
        return generate_synthetic_energy_data(save_path=str(path))
    
    df = pd.read_csv(path, parse_dates=[DATETIME_COL])
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for forecasting.
    
    Features:
    - Lag features (past demand)
    - Rolling statistics
    - Time cyclical encoding
    """
    df = df.copy()
    
    # Lag features
    for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
        df[f'demand_lag_{lag}'] = df[TARGET_COL].shift(lag)
    
    # Rolling statistics
    df['demand_rolling_mean_24'] = df[TARGET_COL].rolling(24).mean()
    df['demand_rolling_std_24'] = df[TARGET_COL].rolling(24).std()
    df['demand_rolling_mean_168'] = df[TARGET_COL].rolling(168).mean()
    
    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Temperature deviation from comfortable range
    df['temp_deviation'] = np.abs(df['temperature'] - 20)
    
    return df


def train_test_split_timeseries(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VALIDATION_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data chronologically.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    logger.info(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    return train, val, test


def prepare_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int = LSTM_SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Feature array (n_samples, n_features)
        target: Target array (n_samples,)
        seq_length: Length of input sequence
        
    Returns:
        X: (n_sequences, seq_length, n_features)
        y: (n_sequences,)
    """
    X, y = [], []
    
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(target[i])
    
    return np.array(X), np.array(y)


class DataNormalizer:
    """Min-Max normalization for time series data."""
    
    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        self.target_min = None
        self.target_max = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        
        if y is not None:
            self.target_min = y.min()
            self.target_max = y.max()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        return (X - self.min_vals) / range_vals
    
    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_min) / (self.target_max - self.target_min)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        return y * (self.target_max - self.target_min) + self.target_min


if __name__ == "__main__":
    # Generate and save data
    df = generate_synthetic_energy_data(save_path=str(DATA_FILE))
    print(f"\nGenerated {len(df)} samples")
    print(f"\nSample data:\n{df.head()}")
    print(f"\nDemand statistics:\n{df[TARGET_COL].describe()}")
