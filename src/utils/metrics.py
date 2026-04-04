"""
Evaluation Metrics for Time Series
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for time series predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    metrics : List[str], optional
        List of metrics to compute. Default: all metrics.
        
    Returns
    -------
    results : Dict[str, float]
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ["mae", "mse", "rmse", "mape", "smape", "r2"]
    
    results = {}
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {m: np.nan for m in metrics}
    
    if "mae" in metrics:
        results["mae"] = mean_absolute_error(y_true, y_pred)
    
    if "mse" in metrics:
        results["mse"] = mean_squared_error(y_true, y_pred)
    
    if "rmse" in metrics:
        results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if "mape" in metrics:
        # Handle zero values
        try:
            results["mape"] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            results["mape"] = np.nan
    
    if "smape" in metrics:
        # Symmetric MAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
        results["smape"] = smape if not np.isnan(smape) else np.inf
    
    if "r2" in metrics:
        results["r2"] = r2_score(y_true, y_pred)
    
    return results


def evaluate_model(
    model,
    y_train: pd.Series,
    y_test: pd.Series,
    steps: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on train and test sets.
    
    Parameters
    ----------
    model
        Fitted model with predict method
    y_train : pd.Series
        Training data
    y_test : pd.Series
        Test data
    steps : int, optional
        Number of steps to predict
        
    Returns
    -------
    results : Dict[str, Dict[str, float]]
        Dictionary with train and test metrics
    """
    if steps is None:
        steps = len(y_test)
    
    # Train predictions
    try:
        y_train_pred = model.predict(steps=len(y_train))
        train_metrics = evaluate_predictions(
            y_train.values, 
            y_train_pred.values if hasattr(y_train_pred, 'values') else y_train_pred
        )
    except:
        train_metrics = {"error": "Could not predict on training data"}
    
    # Test predictions
    try:
        y_test_pred = model.predict(steps=steps)
        test_metrics = evaluate_predictions(
            y_test.values,
            y_test_pred.values if hasattr(y_test_pred, 'values') else y_test_pred
        )
    except:
        test_metrics = {"error": "Could not predict on test data"}
    
    return {
        "train": train_metrics,
        "test": test_metrics
    }


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Pretty print metrics."""
    print(f"\n{title}")
    print("-" * 40)
    for name, value in metrics.items():
        if not np.isnan(value):
            print(f"{name.upper():>10}: {value:.4f}")
        else:
            print(f"{name.upper():>10}: N/A")
