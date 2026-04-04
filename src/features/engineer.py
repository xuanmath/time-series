"""
Feature Engineering for Time Series
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from scipy import stats


class FeatureEngineer:
    """
    Feature engineering for time series data.
    
    Generates various time-based and statistical features.
    """
    
    def __init__(self):
        self.feature_names = []
        
    def create_time_features(
        self, 
        df: pd.DataFrame, 
        time_col: str = "date"
    ) -> pd.DataFrame:
        """
        Create time-based features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        time_col : str
            Name of time column
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame with time features
        """
        df = df.copy()
        dt = pd.to_datetime(df[time_col])
        
        # Basic time features
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["dayofweek"] = dt.dt.dayofweek
        df["dayofyear"] = dt.dt.dayofyear
        df["weekofyear"] = dt.dt.isocalendar().week
        df["quarter"] = dt.dt.quarter
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df["is_month_start"] = dt.dt.is_month_start.astype(int)
        df["is_month_end"] = dt.dt.is_month_end.astype(int)
        
        # Cyclical features
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
        df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        lags: List[int] = [1, 7, 14, 28]
    ) -> pd.DataFrame:
        """
        Create lag features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Target column name
        lags : List[int]
            List of lag periods
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
            
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        windows: List[int] = [7, 14, 28],
        statistics: List[str] = ["mean", "std", "min", "max"]
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Target column name
        windows : List[int]
            List of window sizes
        statistics : List[str]
            List of statistics to compute
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in windows:
            rolling = df[target_col].rolling(window=window)
            
            if "mean" in statistics:
                df[f"{target_col}_rolling_mean_{window}"] = rolling.mean()
            if "std" in statistics:
                df[f"{target_col}_rolling_std_{window}"] = rolling.std()
            if "min" in statistics:
                df[f"{target_col}_rolling_min_{window}"] = rolling.min()
            if "max" in statistics:
                df[f"{target_col}_rolling_max_{window}"] = rolling.max()
            if "skew" in statistics:
                df[f"{target_col}_rolling_skew_{window}"] = rolling.skew()
            if "kurt" in statistics:
                df[f"{target_col}_rolling_kurt_{window}"] = rolling.kurt()
                
        return df
    
    def create_diff_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        periods: List[int] = [1, 7]
    ) -> pd.DataFrame:
        """
        Create difference features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Target column name
        periods : List[int]
            List of difference periods
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame with difference features
        """
        df = df.copy()
        
        for period in periods:
            df[f"{target_col}_diff_{period}"] = df[target_col].diff(period)
            df[f"{target_col}_pct_change_{period}"] = df[target_col].pct_change(period)
            
        return df
    
    def create_ewm_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        spans: List[int] = [7, 14, 28]
    ) -> pd.DataFrame:
        """
        Create exponential weighted mean features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        target_col : str
            Target column name
        spans : List[int]
            List of span values
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame with EWM features
        """
        df = df.copy()
        
        for span in spans:
            df[f"{target_col}_ewm_{span}"] = df[target_col].ewm(span=span).mean()
            
        return df
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create all features at once.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        time_col : str
            Time column name
        target_col : str
            Target column name
        lags : List[int], optional
            Lag periods
        windows : List[int], optional
            Rolling window sizes
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame with all features
        """
        lags = lags or [1, 7, 14, 28]
        windows = windows or [7, 14, 28]
        
        df = self.create_time_features(df, time_col)
        df = self.create_lag_features(df, target_col, lags)
        df = self.create_rolling_features(df, target_col, windows)
        df = self.create_diff_features(df, target_col, [1, 7])
        df = self.create_ewm_features(df, target_col, windows)
        
        return df
