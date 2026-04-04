"""
LSTM Model for Time Series Forecasting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataset(Dataset):
    """Dataset for time series with sliding window."""
    
    def __init__(
        self, 
        data: np.ndarray, 
        seq_len: int,
        pred_len: int = 1
    ):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y


class LSTMNet(nn.Module):
    """LSTM Network for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModel:
    """
    LSTM model wrapper for time series forecasting.
    
    Parameters
    ----------
    seq_len : int
        Input sequence length
    pred_len : int
        Prediction length
    hidden_size : int
        LSTM hidden size
    num_layers : int
        Number of LSTM layers
    dropout : float
        Dropout rate
    bidirectional : bool
        Whether to use bidirectional LSTM
    learning_rate : float
        Learning rate
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    device : str
        'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        seq_len: int = 24,
        pred_len: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "auto"
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.scaler = MinMaxScaler()
        self.input_size = 1
        
    def _build_model(self) -> LSTMNet:
        """Build LSTM model."""
        return LSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.pred_len,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
    
    def fit(
        self, 
        y: pd.Series,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10,
        verbose: int = 1
    ) -> "LSTMModel":
        """
        Fit LSTM model.
        
        Parameters
        ----------
        y : pd.Series
            Time series data
        validation_split : float
            Fraction of data for validation
        early_stopping_patience : int
            Patience for early stopping
        verbose : int
            Verbosity level
            
        Returns
        -------
        self : LSTMModel
            Fitted model
        """
        # Prepare data
        data = y.values.reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data)
        
        # Update input size if multivariate
        self.input_size = data_scaled.shape[1]
        self.model = self._build_model()
        
        # Create dataset
        dataset = TimeSeriesDataset(
            data_scaled, 
            self.seq_len, 
            self.pred_len
        )
        
        # Split train/val
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y.squeeze(-1))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    output = self.model(batch_x)
                    val_loss += criterion(output, batch_y.squeeze(-1)).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose > 0:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        return self
    
    def predict(self, steps: int, last_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        last_values : np.ndarray, optional
            Last known values (seq_len values). If None, uses training data.
            
        Returns
        -------
        predictions : np.ndarray
            Forecast values
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        predictions = []
        
        if last_values is not None:
            current_seq = self.scaler.transform(last_values.reshape(-1, 1))
        else:
            # This should be set during fit
            raise ValueError("Please provide last_values")
        
        current_seq = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(current_seq)
                pred_numpy = pred.cpu().numpy()
                pred_original = self.scaler.inverse_transform(pred_numpy)
                predictions.append(pred_original[0, 0])
                
                # Update sequence
                new_seq = torch.cat([
                    current_seq[:, 1:, :],
                    pred.unsqueeze(-1)
                ], dim=1)
                current_seq = new_seq
        
        return np.array(predictions)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'params': self.get_params()
        }, path)
        
    def load(self, path: str) -> "LSTMModel":
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.scaler = checkpoint['scaler']
        params = checkpoint['params']
        self.seq_len = params['seq_len']
        self.pred_len = params['pred_len']
        self.hidden_size = params['hidden_size']
        self.num_layers = params['num_layers']
        self.input_size = params.get('input_size', 1)
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "input_size": self.input_size
        }
