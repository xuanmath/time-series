"""
GRU Model for Time Series Forecasting
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
import joblib
from pathlib import Path


class GRUModel(nn.Module):
    """
    GRU model for time series forecasting.
    
    Parameters
    ----------
    seq_len : int
        Input sequence length
    pred_len : int
        Prediction length
    hidden_size : int
        Hidden state size
    num_layers : int
        Number of GRU layers
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        seq_len: int = 24,
        pred_len: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, features)
            
        Returns
        -------
        torch.Tensor
            Output of shape (batch, pred_len)
        """
        # GRU forward
        output, _ = self.gru(x)  # output: (batch, seq_len, hidden_size)
        
        # Take last output
        last_output = output[:, -1, :]  # (batch, hidden_size)
        
        # Prediction
        prediction = self.fc(last_output)  # (batch, pred_len)
        
        return prediction


class GRUForecaster:
    """
    GRU Forecaster wrapper with training and prediction.
    """
    
    def __init__(
        self,
        seq_len: int = 24,
        pred_len: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = None
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler = None
        self.train_losses = []
        self.val_losses = []
    
    def _create_sequences(
        self,
        data: np.ndarray,
        seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        verbose: int = 1,
        early_stopping_patience: int = 10
    ):
        """
        Train the GRU model.
        
        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples,) or (n_samples, n_features)
        y : np.ndarray, optional
            Target data (not used, for API consistency)
        verbose : int
            Print progress
        early_stopping_patience : int
            Patience for early stopping
        """
        # Handle input
        if isinstance(X, np.ndarray):
            data = X
        else:
            data = X.values
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(data, self.seq_len)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        self.model = GRUModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if verbose > 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_state_dict = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose > 0:
                        print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_state_dict)
                    break
    
    def predict(
        self,
        X: Optional[np.ndarray] = None,
        steps: Optional[int] = None,
        last_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray, optional
            Input sequences (for multi-step prediction)
        steps : int, optional
            Number of steps to predict
        last_values : np.ndarray, optional
            Last known values for autoregressive prediction
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        self.model.eval()
        
        if X is not None:
            # Direct prediction from sequences
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], X.shape[1], 1)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy()
            
            return predictions.flatten()
        
        elif last_values is not None and steps is not None:
            # Autoregressive prediction
            predictions = []
            current_seq = last_values[-self.seq_len:].copy()
            
            with torch.no_grad():
                for _ in range(steps):
                    seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
                    pred = self.model(seq_tensor).cpu().numpy()[0, 0]
                    predictions.append(pred)
                    
                    # Update sequence
                    current_seq = np.roll(current_seq, -1)
                    current_seq[-1] = pred
            
            return np.array(predictions)
        
        else:
            raise ValueError("Either X or (steps and last_values) must be provided")
    
    def save(self, filepath: str):
        """Save model to file."""
        save_dict = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': {
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'batch_size': self.batch_size
            },
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        save_dict = torch.load(filepath, map_location=self.device)
        
        config = save_dict['config']
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        self.model = GRUModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        if save_dict['model_state_dict']:
            self.model.load_state_dict(save_dict['model_state_dict'])
        
        self.train_losses = save_dict.get('train_losses', [])
        self.val_losses = save_dict.get('val_losses', [])
