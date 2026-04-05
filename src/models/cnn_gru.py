"""
CNN-GRU Model for Time Series Forecasting

CNN extracts local patterns/features, GRU captures temporal dependencies.
Architecture: Input -> CNN(Conv1D) -> GRU -> Output
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Dict, Any
import joblib
from pathlib import Path
from datetime import datetime


class CNNGRUModel(nn.Module):
    """
    CNN-GRU hybrid model for time series forecasting.
    
    Architecture:
        1. CNN layers extract local temporal patterns
        2. GRU layers capture long-term dependencies
        3. Fully connected layers for prediction
    
    Parameters
    ----------
    seq_len : int
        Input sequence length
    pred_len : int
        Prediction length  
    cnn_channels : list
        Number of channels for each CNN layer
    cnn_kernel_sizes : list
        Kernel sizes for each CNN layer
    gru_hidden_size : int
        GRU hidden state size
    gru_num_layers : int
        Number of GRU layers
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        seq_len: int = 24,
        pred_len: int = 1,
        input_features: int = 1,
        cnn_channels: list = [32, 64],
        cnn_kernel_sizes: list = [3, 3],
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_features = input_features
        
        # Validate CNN configuration
        assert len(cnn_channels) == len(cnn_kernel_sizes), \
            "CNN channels and kernel sizes must have same length"
        
        # Build CNN layers
        cnn_layers = []
        in_channels = input_features
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, cnn_kernel_sizes)):
            # Conv1D layer
            cnn_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Preserve sequence length
            ))
            
            # Batch normalization
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm1d(out_channels))
            
            # Activation
            cnn_layers.append(nn.ReLU())
            
            # Dropout
            if dropout > 0:
                cnn_layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_output_channels = cnn_channels[-1]
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=self.cnn_output_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0
        )
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size // 2, pred_len)
        )
    
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
        # CNN expects (batch, channels, seq_len)
        # Input is (batch, seq_len, features)
        x_cnn = x.permute(0, 2, 1)  # (batch, features, seq_len)
        
        # CNN feature extraction
        cnn_out = self.cnn(x_cnn)  # (batch, cnn_channels, seq_len)
        
        # Back to (batch, seq_len, channels) for GRU
        gru_input = cnn_out.permute(0, 2, 1)  # (batch, seq_len, cnn_channels)
        
        # GRU forward
        gru_out, _ = self.gru(gru_input)  # (batch, seq_len, hidden_size)
        
        # Take last output
        last_out = gru_out[:, -1, :]  # (batch, hidden_size)
        
        # Final prediction
        prediction = self.fc(last_out)  # (batch, pred_len)
        
        return prediction


class CNNGRUForecaster:
    """
    CNN-GRU Forecaster wrapper with training, prediction and optimization.
    """
    
    def __init__(
        self,
        seq_len: int = 24,
        pred_len: int = 1,
        input_features: int = 1,
        cnn_channels: list = [32, 64],
        cnn_kernel_sizes: list = [3, 3],
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        weight_decay: float = 1e-5,
        device: str = None
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_features = input_features
        self.cnn_channels = cnn_channels
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.best_state_dict = None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def _create_sequences(
        self,
        data: np.ndarray,
        seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series training."""
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)
    
    def _create_multivariate_sequences(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for multivariate input."""
        X_seq, y_seq = [], []
        for i in range(len(X_data) - seq_len):
            X_seq.append(X_data[i:i + seq_len])
            y_seq.append(y_data[i + seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        verbose: int = 1,
        early_stopping_patience: int = 15,
        validation_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train the CNN-GRU model.
        
        Parameters
        ----------
        X : np.ndarray
            Training data
            - Univariate: shape (n_samples,)
            - Multivariate: shape (n_samples, n_features)
        y : np.ndarray, optional
            Target data (if different from X)
        verbose : int
            Print progress (0: silent, 1: progress, 2: detailed)
        early_stopping_patience : int
            Patience for early stopping
        validation_split : float
            Validation data proportion
            
        Returns
        -------
        dict
            Training history and best metrics
        """
        # Handle input
        if isinstance(X, np.ndarray):
            data = X
        else:
            data = X.values
        
        # Prepare sequences
        if y is not None:
            # Multivariate case
            X_seq, y_seq = self._create_multivariate_sequences(data, y, self.seq_len)
            self.input_features = data.shape[1] if len(data.shape) > 1 else 1
        else:
            # Univariate case
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            X_seq, y_seq = self._create_sequences(data, self.seq_len)
            self.input_features = 1
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # Create DataLoader with validation split
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int((1 - validation_split) * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        self.model = CNNGRUModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            input_features=self.input_features,
            cnn_channels=self.cnn_channels,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            gru_hidden_size=self.gru_hidden_size,
            gru_num_layers=self.gru_num_layers,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        if verbose > 0:
            print(f"Model architecture:")
            print(f"  CNN channels: {self.cnn_channels}")
            print(f"  GRU hidden: {self.gru_hidden_size}, layers: {self.gru_num_layers}")
            print(f"  Device: {self.device}")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2
        )
        
        # Training loop
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            scheduler.step()
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dict = self.model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress output
            if verbose > 0 and (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
                      f"Best: {self.best_val_loss:.6f} @ epoch {best_epoch}, "
                      f"LR: {lr:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose > 0:
                    print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
                break
        
        # Restore best model
        if self.best_state_dict:
            self.model.load_state_dict(self.best_state_dict)
        
        return {
            "best_epoch": best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_epochs": epoch + 1,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
    
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
            Input sequences for direct prediction
        steps : int, optional  
            Number of steps for autoregressive prediction
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
                X = X.reshape(X.shape[0], X.shape[1], self.input_features)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy()
            
            return predictions.flatten()
        
        elif last_values is not None and steps is not None:
            # Autoregressive multi-step prediction
            predictions = []
            current_seq = last_values[-self.seq_len:].copy()
            
            if len(current_seq.shape) == 1:
                current_seq = current_seq.reshape(-1, 1)
            
            with torch.no_grad():
                for _ in range(steps):
                    seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
                    pred = self.model(seq_tensor).cpu().numpy()[0, 0]
                    predictions.append(pred)
                    
                    # Update sequence (slide window)
                    current_seq = np.roll(current_seq, -1, axis=0)
                    current_seq[-1] = pred
            
            return np.array(predictions)
        
        else:
            raise ValueError("Either X or (steps and last_values) must be provided")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns
        -------
        dict
            Metrics: MAE, MSE, RMSE, R2, SMAPE
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Remove NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # SMAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "smape": smape if not np.isnan(smape) else 100.0
        }
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        save_dict = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': {
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'input_features': self.input_features,
                'cnn_channels': self.cnn_channels,
                'cnn_kernel_sizes': self.cnn_kernel_sizes,
                'gru_hidden_size': self.gru_hidden_size,
                'gru_num_layers': self.gru_num_layers,
                'dropout': self.dropout,
                'use_batch_norm': self.use_batch_norm,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'weight_decay': self.weight_decay
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            }
        }
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from checkpoint."""
        save_dict = torch.load(filepath, map_location=self.device)
        
        config = save_dict['config']
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.input_features = config['input_features']
        self.cnn_channels = config['cnn_channels']
        self.cnn_kernel_sizes = config['cnn_kernel_sizes']
        self.gru_hidden_size = config['gru_hidden_size']
        self.gru_num_layers = config['gru_num_layers']
        self.dropout = config['dropout']
        self.use_batch_norm = config['use_batch_norm']
        
        self.model = CNNGRUModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            input_features=self.input_features,
            cnn_channels=self.cnn_channels,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            gru_hidden_size=self.gru_hidden_size,
            gru_num_layers=self.gru_num_layers,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        if save_dict['model_state_dict']:
            self.model.load_state_dict(save_dict['model_state_dict'])
        
        self.train_losses = save_dict.get('training_history', {}).get('train_losses', [])
        self.val_losses = save_dict.get('training_history', {}).get('val_losses', [])
        self.best_val_loss = save_dict.get('training_history', {}).get('best_val_loss', float('inf'))
        
        print(f"Model loaded from {filepath}")


def get_default_config() -> Dict[str, Any]:
    """Get default CNN-GRU configuration."""
    return {
        "seq_len": 24,
        "pred_len": 1,
        "cnn_channels": [32, 64],
        "cnn_kernel_sizes": [3, 3],
        "gru_hidden_size": 64,
        "gru_num_layers": 2,
        "dropout": 0.1,
        "use_batch_norm": True,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "early_stopping_patience": 15
    }


def get_advanced_config() -> Dict[str, Any]:
    """Get advanced CNN-GRU configuration for better performance."""
    return {
        "seq_len": 48,
        "pred_len": 1,
        "cnn_channels": [64, 128, 256],
        "cnn_kernel_sizes": [3, 3, 3],
        "gru_hidden_size": 128,
        "gru_num_layers": 3,
        "dropout": 0.15,
        "use_batch_norm": True,
        "learning_rate": 0.0005,
        "epochs": 200,
        "batch_size": 64,
        "weight_decay": 1e-4,
        "early_stopping_patience": 20
    }