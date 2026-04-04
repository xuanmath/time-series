"""
Data Preprocessor for Wind Power Forecasting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, List, Dict
import joblib
from pathlib import Path


class WindPowerDataProcessor:
    """
    Data processor for wind power forecasting.
    
    Handles:
    - Time feature extraction
    - Feature scaling
    - Train/test split
    - Sequence generation for deep learning
    """
    
    def __init__(
        self,
        target_col: str = "Power",
        time_col: str = "Time",
        feature_cols: Optional[List[str]] = None,
        scaler_type: str = "minmax"
    ):
        self.target_col = target_col
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.scaler_type = scaler_type
        
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from file."""
        df = pd.read_csv(filepath)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        df = df.copy()
        dt = df[self.time_col]
        
        # Basic time features
        df["hour"] = dt.dt.hour
        df["day"] = dt.dt.day
        df["month"] = dt.dt.month
        df["dayofweek"] = dt.dt.dayofweek
        df["dayofyear"] = dt.dt.dayofyear
        
        # Cyclical features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        
        return df
    
    def create_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create wind-specific features."""
        df = df.copy()
        
        # Wind speed features
        if "windspeed_10m" in df.columns and "windspeed_100m" in df.columns:
            df["windspeed_diff"] = df["windspeed_100m"] - df["windspeed_10m"]
            df["windspeed_ratio"] = df["windspeed_100m"] / (df["windspeed_10m"] + 0.01)
        
        # Wind direction features
        if "winddirection_10m" in df.columns and "winddirection_100m" in df.columns:
            df["winddirection_diff"] = df["winddirection_100m"] - df["winddirection_10m"]
            
            # Convert to sin/cos for cyclical nature
            df["winddirection_10m_sin"] = np.sin(np.radians(df["winddirection_10m"]))
            df["winddirection_10m_cos"] = np.cos(np.radians(df["winddirection_10m"]))
            df["winddirection_100m_sin"] = np.sin(np.radians(df["winddirection_100m"]))
            df["winddirection_100m_cos"] = np.cos(np.radians(df["winddirection_100m"]))
        
        # Temperature features
        if "temperature_2m" in df.columns and "dewpoint_2m" in df.columns:
            df["temp_dewpoint_diff"] = df["temperature_2m"] - df["dewpoint_2m"]
        
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        add_time_features: bool = True,
        add_wind_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scalers and transform data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        add_time_features : bool
            Whether to add time features
        add_wind_features : bool
            Whether to add wind-specific features
            
        Returns
        -------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        """
        df = df.copy()
        
        # Add features
        if add_time_features:
            df = self.extract_time_features(df)
        if add_wind_features:
            df = self.create_wind_features(df)
        
        # Remove time column
        df = df.drop(columns=[self.time_col])
        
        # Determine feature columns
        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c != self.target_col]
        
        self.feature_names = self.feature_cols
        
        # Extract features and target
        X = df[self.feature_cols].values
        y = df[self.target_col].values.reshape(-1, 1)
        
        # Initialize scalers
        if self.scaler_type == "minmax":
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        else:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        
        # Fit and transform
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        return X_scaled, y_scaled.flatten()
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform new data using fitted scalers."""
        df = df.copy()
        df = self.extract_time_features(df)
        df = self.create_wind_features(df)
        df = df.drop(columns=[self.time_col])
        
        X = df[self.feature_cols].values
        y = df[self.target_col].values.reshape(-1, 1)
        
        X_scaled = self.feature_scaler.transform(X)
        y_scaled = self.target_scaler.transform(y)
        
        return X_scaled, y_scaled.flatten()
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform target values."""
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_scaled).flatten()
    
    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int,
        pred_len: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for sequence models (LSTM, Transformer).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        seq_len : int
            Input sequence length
        pred_len : int
            Prediction length
            
        Returns
        -------
        X_seq : np.ndarray
            Input sequences (n_samples, seq_len, n_features)
        y_seq : np.ndarray
            Target sequences (n_samples, pred_len)
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - seq_len - pred_len + 1):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len:i + seq_len + pred_len])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        shuffle: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        For time series, default is no shuffle to preserve temporal order.
        """
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        if shuffle:
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
        
        X_train = X[:-n_test]
        X_test = X[-n_test:]
        y_train = y[:-n_test]
        y_test = y[-n_test:]
        
        return X_train, X_test, y_train, y_test
    
    def save(self, path: str) -> None:
        """Save processor state."""
        state = {
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_cols": self.feature_cols,
            "feature_names": self.feature_names,
            "target_col": self.target_col,
            "time_col": self.time_col,
            "scaler_type": self.scaler_type
        }
        joblib.dump(state, path)
        
    def load(self, path: str) -> "WindPowerDataProcessor":
        """Load processor state."""
        state = joblib.load(path)
        self.feature_scaler = state["feature_scaler"]
        self.target_scaler = state["target_scaler"]
        self.feature_cols = state["feature_cols"]
        self.feature_names = state["feature_names"]
        self.target_col = state["target_col"]
        self.time_col = state["time_col"]
        self.scaler_type = state["scaler_type"]
        return self
