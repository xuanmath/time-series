"""
Unit tests for time series models
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ARIMAModel, LSTMModel
from src.features import FeatureEngineer
from src.utils.metrics import evaluate_predictions
from src.utils.data import generate_sample_data


class TestARIMAModel:
    """Tests for ARIMA model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        df = generate_sample_data(n_samples=100)
        return df
    
    def test_arima_fit(self, sample_data):
        """Test ARIMA fitting."""
        model = ARIMAModel(order=(1, 1, 1))
        y = sample_data.set_index("date")["value"]
        
        model.fit(y)
        
        assert model.fitted_model is not None
    
    def test_arima_predict(self, sample_data):
        """Test ARIMA prediction."""
        model = ARIMAModel(order=(1, 1, 1))
        y = sample_data.set_index("date")["value"]
        
        model.fit(y)
        predictions = model.predict(steps=10)
        
        assert len(predictions) == 10
    
    def test_arima_get_params(self, sample_data):
        """Test ARIMA get_params."""
        model = ARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        
        params = model.get_params()
        
        assert params["order"] == (1, 1, 1)
        assert params["seasonal_order"] == (1, 1, 1, 12)


class TestLSTMModel:
    """Tests for LSTM model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        df = generate_sample_data(n_samples=200)
        return df
    
    def test_lstm_init(self):
        """Test LSTM initialization."""
        model = LSTMModel(
            seq_len=10,
            hidden_size=32,
            num_layers=1,
            epochs=2
        )
        
        assert model.seq_len == 10
        assert model.hidden_size == 32
    
    def test_lstm_fit(self, sample_data):
        """Test LSTM fitting."""
        model = LSTMModel(
            seq_len=10,
            hidden_size=16,
            num_layers=1,
            epochs=2,
            verbose=0
        )
        y = sample_data.set_index("date")["value"]
        
        model.fit(y)
        
        assert model.model is not None


class TestFeatureEngineer:
    """Tests for feature engineering."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        df = generate_sample_data(n_samples=100)
        return df
    
    def test_time_features(self, sample_data):
        """Test time feature creation."""
        engineer = FeatureEngineer()
        
        df = engineer.create_time_features(sample_data, "date")
        
        assert "year" in df.columns
        assert "month" in df.columns
        assert "day" in df.columns
        assert "dayofweek" in df.columns
    
    def test_lag_features(self, sample_data):
        """Test lag feature creation."""
        engineer = FeatureEngineer()
        
        df = engineer.create_lag_features(sample_data, "value", lags=[1, 7])
        
        assert "value_lag_1" in df.columns
        assert "value_lag_7" in df.columns
    
    def test_rolling_features(self, sample_data):
        """Test rolling feature creation."""
        engineer = FeatureEngineer()
        
        df = engineer.create_rolling_features(
            sample_data, 
            "value", 
            windows=[7],
            statistics=["mean", "std"]
        )
        
        assert "value_rolling_mean_7" in df.columns
        assert "value_rolling_std_7" in df.columns


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_mae(self):
        """Test MAE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        
        metrics = evaluate_predictions(y_true, y_pred)
        
        assert "mae" in metrics
        assert metrics["mae"] > 0
    
    def test_rmse(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        metrics = evaluate_predictions(y_true, y_pred)
        
        assert metrics["rmse"] == 0.0
    
    def test_r2(self):
        """Test R² calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluate_predictions(y_true, y_pred)
        
        assert "r2" in metrics
        assert 0 < metrics["r2"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
