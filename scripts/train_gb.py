"""
Wind Power Forecasting with Gradient Boosting (No PyTorch required)
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import WindPowerDataProcessor
from src.utils.metrics import evaluate_predictions, print_metrics

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_features_for_ml(
    X: np.ndarray, 
    y: np.ndarray, 
    seq_len: int = 24
) -> tuple:
    """
    Create features for ML models by adding lagged values.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    seq_len : int
        Number of lagged features to create
        
    Returns
    -------
    X_new : np.ndarray
        Extended feature matrix with lagged values
    y_new : np.ndarray
        Target values (shifted)
    """
    n_samples = len(X) - seq_len
    n_features = X.shape[1]
    
    X_new = np.zeros((n_samples, n_features * (seq_len + 1)))
    
    for i in range(n_samples):
        # Current features
        X_new[i, :n_features] = X[i + seq_len]
        
        # Lagged features
        for j in range(seq_len):
            start_idx = n_features + j * n_features
            X_new[i, start_idx:start_idx + n_features] = X[i + seq_len - 1 - j]
        
        # Lagged target values as features
        # (handled separately below)
    
    y_new = y[seq_len:]
    
    # Add lagged target values
    X_with_y = np.hstack([X_new, np.zeros((n_samples, seq_len))])
    for i in range(n_samples):
        for j in range(seq_len):
            X_with_y[i, -seq_len + j] = y[i + seq_len - 1 - j]
    
    return X_with_y, y_new


def train_xgboost(X_train, y_train, X_test, y_test, config):
    """Train XGBoost model."""
    model_config = config["model"].get("xgboost", {})
    
    model = xgb.XGBRegressor(
        n_estimators=model_config.get("n_estimators", 100),
        max_depth=model_config.get("max_depth", 6),
        learning_rate=model_config.get("learning_rate", 0.1),
        subsample=model_config.get("subsample", 0.8),
        colsample_bytree=model_config.get("colsample_bytree", 0.8),
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, config):
    """Train LightGBM model."""
    model_config = config["model"].get("lightgbm", {})
    
    model = lgb.LGBMRegressor(
        n_estimators=model_config.get("n_estimators", 100),
        max_depth=model_config.get("max_depth", 6),
        learning_rate=model_config.get("learning_rate", 0.1),
        num_leaves=model_config.get("num_leaves", 31),
        subsample=model_config.get("subsample", 0.8),
        colsample_bytree=model_config.get("colsample_bytree", 0.8),
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(0)]
    )
    
    return model


def train_gradient_boosting(X_train, y_train, X_test, y_test, config):
    """Train sklearn GradientBoosting model."""
    model_config = config["model"].get("gradient_boosting", {})
    
    model = GradientBoostingRegressor(
        n_estimators=model_config.get("n_estimators", 100),
        max_depth=model_config.get("max_depth", 4),
        learning_rate=model_config.get("learning_rate", 0.1),
        subsample=model_config.get("subsample", 0.8),
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Wind Power Forecasting")
    parser.add_argument("--config", type=str, default="configs/wind_power.yaml", help="Config file path")
    parser.add_argument("--model", type=str, choices=["xgboost", "lightgbm", "gradient_boosting"], default="lightgbm", help="Model type")
    parser.add_argument("--seq-len", type=int, default=24, help="Sequence length for lagged features")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config["model"]["name"] = args.model
    
    print("=" * 60)
    print(f"Wind Power Forecasting - {args.model.upper()}")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    processor = WindPowerDataProcessor(
        target_col="Power",
        time_col="Time",
        scaler_type="minmax"
    )
    
    df = processor.load_data(config["data"]["filepath"])
    print(f"    Data shape: {df.shape}")
    print(f"    Time range: {df['Time'].min()} ~ {df['Time'].max()}")
    
    # Preprocess
    print("\n[2/4] Preprocessing...")
    X, y = processor.fit_transform(df, add_time_features=True, add_wind_features=True)
    print(f"    Features: {X.shape[1]}")
    print(f"    Samples: {len(X)}")
    
    # Create lagged features
    print(f"    Creating lagged features (seq_len={args.seq_len})...")
    X_with_lags, y_shifted = create_features_for_ml(X, y, args.seq_len)
    print(f"    Extended features: {X_with_lags.shape[1]}")
    
    # Split
    test_size = 0.2
    n_test = int(len(X_with_lags) * test_size)
    X_train, X_test = X_with_lags[:-n_test], X_with_lags[-n_test:]
    y_train, y_test = y_shifted[:-n_test], y_shifted[-n_test:]
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    print("\n[3/4] Training model...")
    
    if args.model == "xgboost":
        model = train_xgboost(X_train, y_train, X_test, y_test, config)
    elif args.model == "lightgbm":
        model = train_lightgbm(X_train, y_train, X_test, y_test, config)
    else:
        model = train_gradient_boosting(X_train, y_train, X_test, y_test, config)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Inverse transform
    predictions_original = processor.inverse_transform_target(predictions)
    y_test_original = processor.inverse_transform_target(y_test)
    
    # Evaluate
    print("\n[4/4] Evaluating...")
    metrics = evaluate_predictions(y_test_original, predictions_original)
    print_metrics(metrics, title="Test Results")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        top_k = 10
        top_indices = np.argsort(importance)[-top_k:][::-1]
        print(f"\nTop {top_k} features:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {importance[idx]:.4f}")
    
    # Save results
    output_path = Path(config["output"]["results_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    pred_df = pd.DataFrame({
        "actual": y_test_original,
        "predicted": predictions_original,
        "error": y_test_original - predictions_original
    })
    pred_df.to_csv(output_path / f"predictions_{args.model}_{timestamp}.csv", index=False)
    print(f"\nPredictions saved to {output_path / f'predictions_{args.model}_{timestamp}.csv'}")
    
    # Save metrics
    with open(output_path / f"metrics_{args.model}_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
