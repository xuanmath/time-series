"""
Wind Power Forecasting with sklearn (No extra dependencies)
Enhanced with logging
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
from src.utils.metrics import evaluate_predictions
from src.utils.logging_utils import setup_experiment

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import joblib


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_features_for_ml(
    X: np.ndarray, 
    y: np.ndarray, 
    seq_len: int = 24
) -> tuple:
    """Create features for ML models by adding lagged values."""
    n_samples = len(X) - seq_len
    n_features = X.shape[1]
    
    # Current features + lagged features + lagged target
    X_new = np.zeros((n_samples, n_features * (seq_len + 1) + seq_len))
    
    for i in range(n_samples):
        # Current features
        X_new[i, :n_features] = X[i + seq_len]
        
        # Lagged features
        for j in range(seq_len):
            start_idx = n_features + j * n_features
            X_new[i, start_idx:start_idx + n_features] = X[i + seq_len - 1 - j]
        
        # Lagged target values
        for j in range(seq_len):
            X_new[i, -seq_len + j] = y[i + seq_len - 1 - j]
    
    y_new = y[seq_len:]
    
    return X_new, y_new


def main():
    parser = argparse.ArgumentParser(description="Wind Power Forecasting")
    parser.add_argument("--config", type=str, default="configs/wind_power.yaml", help="Config file path")
    parser.add_argument("--model", type=str, choices=["gradient_boosting", "random_forest", "ridge"], default="gradient_boosting", help="Model type")
    parser.add_argument("--seq-len", type=int, default=24, help="Sequence length for lagged features")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--max-depth", type=int, default=6, help="Max depth for trees")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    args = parser.parse_args()
    
    # Model config for logging
    model_config = {
        "model_type": args.model,
        "seq_len": args.seq_len,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate
    }
    
    # Setup logging
    logger = setup_experiment(args.model, model_config, log_dir="logs")
    
    # Load data
    logger.info("\n[1/4] Loading data...")
    processor = WindPowerDataProcessor(
        target_col="Power",
        time_col="Time",
        scaler_type="minmax"
    )
    
    df = processor.load_data("data/raw/wind_power.csv")
    time_range = f"{df['Time'].min()} ~ {df['Time'].max()}"
    logger.info(f"    Data shape: {df.shape}")
    logger.info(f"    Time range: {time_range}")
    
    # Preprocess
    logger.info("\n[2/4] Preprocessing...")
    X, y = processor.fit_transform(df, add_time_features=True, add_wind_features=True)
    logger.info(f"    Original features: {X.shape[1]}")
    logger.info(f"    Total samples: {len(X)}")
    
    # Create lagged features
    logger.info(f"    Creating lagged features (seq_len={args.seq_len})...")
    X_with_lags, y_shifted = create_features_for_ml(X, y, args.seq_len)
    logger.info(f"    Extended features: {X_with_lags.shape[1]}")
    
    # Split
    test_size = 0.2
    n_test = int(len(X_with_lags) * test_size)
    X_train, X_test = X_with_lags[:-n_test], X_with_lags[-n_test:]
    y_train, y_test = y_shifted[:-n_test], y_shifted[-n_test:]
    
    logger.log_data_info(
        n_samples=len(X_with_lags),
        n_features=X_with_lags.shape[1],
        n_train=len(X_train),
        n_test=len(X_test),
        time_range=time_range
    )
    
    # Train
    logger.log_training_start(args.model)
    
    if args.model == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.8,
            random_state=42,
            verbose=0  # We handle logging ourselves
        )
    elif args.model == "random_forest":
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
    else:
        model = Ridge(alpha=1.0)
    
    # Train with timing
    train_start = datetime.now()
    model.fit(X_train, y_train)
    train_duration = (datetime.now() - train_start).total_seconds()
    logger.info(f"    Training time: {train_duration:.2f} seconds")
    
    logger.log_training_complete()
    logger.info("[OK] Model trained successfully")
    
    # Predict
    logger.info("\n[4/4] Evaluating...")
    predictions = model.predict(X_test)
    
    # Inverse transform
    predictions_original = processor.inverse_transform_target(predictions)
    y_test_original = processor.inverse_transform_target(y_test)
    
    # Evaluate
    metrics = evaluate_predictions(y_test_original, predictions_original)
    logger.log_results(metrics)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        importance_dict = dict(zip(feature_names, importance))
        logger.log_feature_importance(importance_dict, top_k=10)
    
    # Save results
    output_path = Path("data/results")
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    pred_file = output_path / f"predictions_{args.model}_{timestamp}.csv"
    pred_df = pd.DataFrame({
        "actual": y_test_original,
        "predicted": predictions_original,
        "error": y_test_original - predictions_original
    })
    pred_df.to_csv(pred_file, index=False)
    logger.log_file_saved("Predictions", str(pred_file))
    
    # Save model
    model_path = Path("models/checkpoints")
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{args.model}_model.pkl"
    processor_file = model_path / "processor.pkl"
    joblib.dump(model, model_file)
    joblib.dump(processor, processor_file)
    logger.log_file_saved("Model", str(model_file))
    logger.log_file_saved("Processor", str(processor_file))
    
    # Finalize logging
    final_metrics = logger.finalize()
    
    logger.log_separator()
    logger.info("[DONE] All completed!")
    logger.log_separator()


if __name__ == "__main__":
    main()
