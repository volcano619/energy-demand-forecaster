"""
Deep Learning Models for Energy Demand Forecasting

Implements:
1. LSTM - Long Short-Term Memory network
2. Simple Transformer encoder (lightweight)
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep learning models will use numpy fallback.")


from config import (
    LSTM_SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE
)


if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """LSTM for time series forecasting."""
        
        def __init__(
            self,
            input_size: int,
            hidden_size: int = LSTM_HIDDEN_SIZE,
            num_layers: int = LSTM_NUM_LAYERS,
            dropout: float = LSTM_DROPOUT,
            output_size: int = 1
        ):
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Take last time step
            last_out = lstm_out[:, -1, :]
            return self.fc(last_out)


class LSTMForecaster:
    """
    LSTM-based forecaster wrapper.
    
    Handles training, prediction, and sequence preparation.
    """
    
    def __init__(
        self,
        input_size: int = None,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        learning_rate: float = LSTM_LEARNING_RATE
    ):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = None
        self.is_fitted = False
        self.device = 'cpu'
        
        # Store normalization parameters
        self.feature_min = None
        self.feature_max = None
        self.target_min = None
        self.target_max = None
        
        # Store last sequence for forecasting
        self.last_sequence = None
    
    def _normalize(self, X: np.ndarray, y: np.ndarray = None, fit: bool = False):
        """Min-max normalization."""
        if fit:
            self.feature_min = X.min(axis=0)
            self.feature_max = X.max(axis=0)
            if y is not None:
                self.target_min = y.min()
                self.target_max = y.max()
        
        range_val = self.feature_max - self.feature_min
        range_val[range_val == 0] = 1
        X_norm = (X - self.feature_min) / range_val
        
        if y is not None:
            y_norm = (y - self.target_min) / (self.target_max - self.target_min + 1e-8)
            return X_norm, y_norm
        return X_norm
    
    def _denormalize_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse normalization for predictions."""
        return y * (self.target_max - self.target_min) + self.target_min
    
    def _create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for LSTM."""
        sequences, targets = [], []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: bool = True
    ) -> 'LSTMForecaster':
        """
        Train LSTM model.
        
        Args:
            X: Features array (n_samples, n_features)
            y: Target array (n_samples,)
            X_val: Validation features
            y_val: Validation targets
            verbose: Print training progress
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using simple fallback.")
            self._fit_fallback(X, y)
            return self
        
        # Normalize
        X_norm, y_norm = self._normalize(X, y, fit=True)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_norm, y_norm)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
        
        self.input_size = X_seq.shape[2]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        # Store last sequence for future predictions
        self.last_sequence = X_norm[-self.sequence_length:]
        self.is_fitted = True
        
        return self
    
    def _fit_fallback(self, X: np.ndarray, y: np.ndarray):
        """Simple fallback when PyTorch unavailable."""
        self.feature_min = X.min(axis=0)
        self.feature_max = X.max(axis=0)
        self.target_min = y.min()
        self.target_max = y.max()
        
        # Store last values for naive prediction
        self._last_values = y[-self.sequence_length:]
        self.is_fitted = True
    
    def predict(self, periods: int = 1, X_future: np.ndarray = None) -> np.ndarray:
        """
        Generate forecasts.
        
        For multi-step forecasting, uses recursive prediction.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not TORCH_AVAILABLE:
            # Fallback: return rolling mean
            return np.full(periods, self._last_values.mean())
        
        self.model.eval()
        predictions = []
        
        # Start with last known sequence
        current_seq = self.last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(periods):
                # Prepare input
                X_input = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
                
                # Predict
                pred = self.model(X_input).cpu().numpy()[0, 0]
                predictions.append(pred)
                
                # Update sequence for next prediction
                new_row = current_seq[-1].copy()
                new_row[0] = pred  # Assume first feature is target (lag-1)
                current_seq = np.vstack([current_seq[1:], new_row])
        
        # Denormalize
        predictions = np.array(predictions)
        return self._denormalize_target(predictions)
    
    def save(self, path: str):
        """Save model to disk."""
        if TORCH_AVAILABLE and self.model:
            torch.save({
                'model_state': self.model.state_dict(),
                'config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers
                },
                'normalization': {
                    'feature_min': self.feature_min,
                    'feature_max': self.feature_max,
                    'target_min': self.target_min,
                    'target_max': self.target_max
                },
                'last_sequence': self.last_sequence
            }, path)
    
    def load(self, path: str):
        """Load model from disk."""
        if TORCH_AVAILABLE:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.input_size = checkpoint['config']['input_size']
            self.hidden_size = checkpoint['config']['hidden_size']
            self.num_layers = checkpoint['config']['num_layers']
            
            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            )
            self.model.load_state_dict(checkpoint['model_state'])
            
            self.feature_min = checkpoint['normalization']['feature_min']
            self.feature_max = checkpoint['normalization']['feature_max']
            self.target_min = checkpoint['normalization']['target_min']
            self.target_max = checkpoint['normalization']['target_max']
            self.last_sequence = checkpoint['last_sequence']
            
            self.is_fitted = True
