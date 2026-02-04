"""
Configuration Module for Energy Demand Forecasting System

Centralizes all configuration parameters.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "saved_models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "energy_data.csv"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Synthetic data parameters
DATA_START_DATE = "2022-01-01"
DATA_END_DATE = "2023-12-31"
FREQUENCY = "H"  # Hourly

# Features
TARGET_COL = "demand_mw"
DATETIME_COL = "timestamp"
FEATURE_COLS = ["temperature", "humidity", "hour", "day_of_week", "month", "is_weekend", "is_holiday"]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Train/Test split
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1

# Forecasting horizons
FORECAST_HORIZONS = {
    "24h": 24,
    "48h": 48,
    "7d": 168
}
DEFAULT_HORIZON = 24

# ============================================================================
# PROPHET CONFIGURATION
# ============================================================================
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_DAILY_SEASONALITY = True
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05

# ============================================================================
# LSTM CONFIGURATION
# ============================================================================
LSTM_SEQUENCE_LENGTH = 168  # 7 days of hourly data
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001

# ============================================================================
# ENSEMBLE CONFIGURATION
# ============================================================================
ENSEMBLE_WEIGHTS = {
    "prophet": 0.4,
    "lstm": 0.6
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
MAPE_TARGET = 5.0  # Target: <5% error
RMSE_ACCEPTABLE_RANGE = 0.1  # Within 10% of mean demand

# ============================================================================
# ANOMALY DETECTION
# ============================================================================
ANOMALY_THRESHOLD_SIGMA = 3.0  # 3 standard deviations

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================
APP_TITLE = "⚡ Energy Demand Forecasting"
APP_LAYOUT = "wide"
DEBUG_MODE = True

# ============================================================================
# VISUALIZATION
# ============================================================================
CHART_HEIGHT = 400
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
ANOMALY_COLOR = "#d62728"
