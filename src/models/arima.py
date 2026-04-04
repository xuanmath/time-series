"""
ARIMA Model for Time Series Forecasting
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Optional, Tuple, Dict, Any
import joblib
from pathlib import Path


class ARIMAModel:
    """
    ARIMA/SARIMA model wrapper for time series forecasting.
    
    Parameters
    ----------
    order : tuple
        (p, d, q) order of ARIMA
    seasonal_order : tuple, optional
        (P, D, Q, s) seasonal order for SARIMA
    """
    
    def __init__(
        self, 
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> "ARIMAModel":
        """
        Fit ARIMA model.
        
        Parameters
        ----------
        y : pd.Series
            Time series data
        exog : pd.DataFrame, optional
            Exogenous variables
            
        Returns
        -------
        self : ARIMAModel
            Fitted model
        """
        if self.seasonal_order:
            self.model = SARIMAX(
                y, 
                exog=exog,
                order=self.order, 
                seasonal_order=self.seasonal_order
            )
        else:
            self.model = ARIMA(y, exog=exog, order=self.order)
            
        self.fitted_model = self.model.fit()
        return self
    
    def predict(
        self, 
        steps: int, 
        exog: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Make predictions.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        exog : pd.DataFrame, optional
            Exogenous variables for forecast period
            
        Returns
        -------
        predictions : pd.Series
            Forecast values
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        predictions = self.fitted_model.forecast(steps=steps, exog=exog)
        return predictions
    
    def get_summary(self) -> str:
        """Get model summary."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return str(self.fitted_model.summary())
    
    def save(self, path: str) -> None:
        """Save model to file."""
        joblib.dump(self.fitted_model, path)
        
    def load(self, path: str) -> "ARIMAModel":
        """Load model from file."""
        self.fitted_model = joblib.load(path)
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order
        }
