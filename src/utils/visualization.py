"""
Visualization Utilities for Time Series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any
from pathlib import Path


def plot_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Forecast vs Actual",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot forecast against actual values.
    
    Parameters
    ----------
    y_true : pd.Series
        Actual values
    y_pred : pd.Series
        Predicted values
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(y_true.index, y_true, label="Actual", color="blue", alpha=0.7)
    ax.plot(y_pred.index, y_pred, label="Predicted", color="red", alpha=0.7)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_decomposition(
    df: pd.DataFrame,
    time_col: str = "date",
    value_col: str = "value",
    period: int = 365,
    figsize: tuple = (12, 10)
) -> plt.Figure:
    """
    Plot time series decomposition.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    time_col : str
        Time column name
    value_col : str
        Value column name
    period : int
        Period for decomposition
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Prepare data
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    
    # Decompose
    decomposition = seasonal_decompose(df[value_col], period=period)
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    decomposition.observed.plot(ax=axes[0], title="Original")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
    decomposition.resid.plot(ax=axes[3], title="Residual")
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    figsize: tuple = (12, 5)
) -> plt.Figure:
    """
    Plot ACF and PACF.
    
    Parameters
    ----------
    series : pd.Series
        Time series data
    lags : int
        Number of lags
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    plot_acf(series, lags=lags, ax=axes[0])
    plot_pacf(series, lags=lags, ax=axes[1])
    
    axes[0].set_title("Autocorrelation")
    axes[1].set_title("Partial Autocorrelation")
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance: Dict[str, float],
    title: str = "Feature Importance",
    figsize: tuple = (10, 6),
    top_n: int = 20
) -> plt.Figure:
    """
    Plot feature importance.
    
    Parameters
    ----------
    importance : Dict[str, float]
        Feature importance dictionary
    title : str
        Plot title
    figsize : tuple
        Figure size
    top_n : int
        Number of top features to show
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    # Sort and select top features
    sorted_importance = sorted(
        importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_n]
    
    features = [x[0] for x in sorted_importance]
    values = [x[1] for x in sorted_importance]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(range(len(features)), values, align="center")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    
    plt.tight_layout()
    return fig
