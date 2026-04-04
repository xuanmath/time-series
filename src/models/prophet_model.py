"""
Prophet Model for Time Series Forecasting
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import joblib

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetModel:
    """
    Facebook Prophet model wrapper for time series forecasting.
    
    Parameters
    ----------
    growth : str
        'linear' or 'logistic' growth trend
    seasonality_mode : str
        'additive' or 'multiplicative'
    yearly_seasonality : bool or int
        Yearly seasonality
    weekly_seasonality : bool or int
        Weekly seasonality
    daily_seasonality : bool or int
        Daily seasonality
    """
    
    def __init__(
        self,
        growth: str = "linear",
        seasonality_mode: str = "additive",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        **kwargs
    ):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Run: pip install prophet")
            
        self.model = Prophet(
            growth=growth,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            **kwargs
        )
        self.fitted = False
        self.params = {
            "growth": growth,
            "seasonality_mode": seasonality_mode,
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            **kwargs
        }
        
    def fit(self, df: pd.DataFrame) -> "ProphetModel":
        """
        Fit Prophet model.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'ds' (date) and 'y' (value) columns
            
        Returns
        -------
        self : ProphetModel
            Fitted model
        """
        self.model.fit(df)
        self.fitted = True
        return self
    
    def predict(
        self, 
        periods: int,
        freq: str = "D",
        include_history: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions.
        
        Parameters
        ----------
        periods : int
            Number of periods to forecast
        freq : str
            Frequency string (e.g., 'D', 'W', 'M')
        include_history : bool
            Whether to include historical data
            
        Returns
        -------
        forecast : pd.DataFrame
            Forecast DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        future = self.model.make_future_dataframe(
            periods=periods, 
            freq=freq,
            include_history=include_history
        )
        forecast = self.model.predict(future)
        return forecast
    
    def add_regressor(
        self, 
        name: str, 
        prior_scale: Optional[float] = None
    ) -> None:
        """Add external regressor."""
        self.model.add_regressor(name, prior_scale=prior_scale)
        
    def add_country_holidays(self, country_name: str) -> None:
        """Add country holidays."""
        self.model.add_country_holidays(country_name=country_name)
        
    def plot(self, forecast: pd.DataFrame, **kwargs):
        """Plot forecast."""
        return self.model.plot(forecast, **kwargs)
    
    def plot_components(self, forecast: pd.DataFrame, **kwargs):
        """Plot forecast components."""
        return self.model.plot_components(forecast, **kwargs)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        joblib.dump(self.model, path)
        
    def load(self, path: str) -> "ProphetModel":
        """Load model from file."""
        self.model = joblib.load(path)
        self.fitted = True
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params
