"""
Transformer Model for Time Series Forecasting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        output_size: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        seq_len: int = 24
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len * 2, dropout=dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(
        self, 
        x: torch.Tensor,
        tgt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, input_size)
        tgt : torch.Tensor, optional
            Target tensor for decoder (batch, seq_len, input_size)
            
        Returns
        -------
        output : torch.Tensor
            Output tensor (batch, pred_len, output_size)
        """
        # Project input to d_model
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        if tgt is not None:
            tgt = self.input_projection(tgt)
            tgt = self.pos_encoder(tgt)
        
        # Transformer
        output = self.transformer(x, tgt if tgt is not None else x)
        
        # Project to output
        output = self.output_projection(output)
        
        return output[:, -1, :]


class TransformerForecaster:
    """
    Transformer model wrapper for time series forecasting.
    
    Parameters
    ----------
    seq_len : int
        Input sequence length
    pred_len : int
        Prediction length
    d_model : int
        Model dimension
    nhead : int
        Number of attention heads
    num_encoder_layers : int
        Number of encoder layers
    num_decoder_layers : int
        Number of decoder layers
    dim_feedforward : int
        Feedforward dimension
    dropout : float
        Dropout rate
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
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "auto"
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.input_size = 1
        
    def _build_model(self) -> TransformerModel:
        """Build Transformer model."""
        return TransformerModel(
            input_size=self.input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_size=self.pred_len,
            seq_len=self.seq_len
        ).to(self.device)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.1,
        early_stopping_patience: int = 15,
        verbose: int = 1
    ) -> "TransformerForecaster":
        """
        Fit Transformer model.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences (n_samples, seq_len, n_features)
        y : np.ndarray
            Target values (n_samples, pred_len)
        validation_split : float
            Fraction of data for validation
        early_stopping_patience : int
            Patience for early stopping
        verbose : int
            Verbosity level
            
        Returns
        -------
        self : TransformerForecaster
            Fitted model
        """
        # Update input size
        self.input_size = X.shape[2]
        self.model = self._build_model()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Split train/val
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        
        X_train, X_val = X_tensor[:-n_val], X_tensor[-n_val:]
        y_train, y_val = y_tensor[:-n_val], y_tensor[-n_val:]
        
        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val)
                val_loss = criterion(val_output, y_val).item()
            
            train_loss /= len(train_loader)
            scheduler.step()
            
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences (n_samples, seq_len, n_features)
            
        Returns
        -------
        predictions : np.ndarray
            Predicted values (n_samples, pred_len)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': self.get_params()
        }, path)
        
    def load(self, path: str) -> "TransformerForecaster":
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        params = checkpoint['params']
        self.seq_len = params['seq_len']
        self.pred_len = params['pred_len']
        self.d_model = params['d_model']
        self.nhead = params['nhead']
        self.num_encoder_layers = params['num_encoder_layers']
        self.num_decoder_layers = params['num_decoder_layers']
        self.dim_feedforward = params['dim_feedforward']
        self.input_size = params['input_size']
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "input_size": self.input_size
        }
