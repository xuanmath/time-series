"""
Data Loading and Saving Utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union


def load_data(
    filepath: str,
    time_col: Optional[str] = None,
    value_col: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load time series data from various file formats.
    
    Parameters
    ----------
    filepath : str
        Path to data file
    time_col : str, optional
        Name of time column
    value_col : str, optional
        Name of value column
    **kwargs
        Additional arguments passed to pd.read_csv/pd.read_parquet
        
    Returns
    -------
    df : pd.DataFrame
        Loaded DataFrame
    """
    path = Path(filepath)
    
    if path.suffix == ".csv":
        df = pd.read_csv(filepath, **kwargs)
    elif path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(filepath, **kwargs)
    elif path.suffix == ".xlsx":
        df = pd.read_excel(filepath, **kwargs)
    elif path.suffix == ".json":
        df = pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Set time column as index if provided
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()
        
    return df


def save_data(
    df: pd.DataFrame,
    filepath: str,
    **kwargs
) -> None:
    """
    Save DataFrame to file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str
        Output file path
    **kwargs
        Additional arguments for saving
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == ".csv":
        df.to_csv(filepath, **kwargs)
    elif path.suffix in [".parquet", ".pq"]:
        df.to_parquet(filepath, **kwargs)
    elif path.suffix == ".xlsx":
        df.to_excel(filepath, **kwargs)
    elif path.suffix == ".json":
        df.to_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def generate_sample_data(
    n_samples: int = 365,
    freq: str = "D",
    start_date: str = "2020-01-01",
    trend: float = 0.1,
    seasonality: float = 10.0,
    noise: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate sample time series data with trend, seasonality, and noise.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    freq : str
        Frequency string
    start_date : str
        Start date
    trend : float
        Trend coefficient
    seasonality : float
        Seasonality amplitude
    noise : float
        Noise standard deviation
    seed : int
        Random seed
        
    Returns
    -------
    df : pd.DataFrame
        Generated DataFrame with 'date' and 'value' columns
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
    t = np.arange(n_samples)
    
    # Trend
    trend_component = trend * t
    
    # Seasonality (yearly)
    seasonality_component = seasonality * np.sin(2 * np.pi * t / 365)
    
    # Noise
    noise_component = np.random.normal(0, noise, n_samples)
    
    # Combine
    values = 100 + trend_component + seasonality_component + noise_component
    
    df = pd.DataFrame({
        "date": dates,
        "value": values
    })
    
    return df
