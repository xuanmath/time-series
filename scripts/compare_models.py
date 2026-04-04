"""
Model Comparison Script for Wind Power Forecasting
Compares: GradientBoosting, RandomForest, LSTM, GRU, Transformer
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import WindPowerDataProcessor
from src.utils.metrics import evaluate_predictions
from src.utils.logging_utils import setup_experiment

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import joblib

# PyTorch models
import torch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_features_for_ml(X: np.ndarray, y: np.ndarray, seq_len: int = 24) -> tuple:
    """Create features for ML models by adding lagged values."""
    n_samples = len(X) - seq_len
    n_features = X.shape[1]
    
    X_new = np.zeros((n_samples, n_features * (seq_len + 1) + seq_len))
    
    for i in range(n_samples):
        X_new[i, :n_features] = X[i + seq_len]
        for j in range(seq_len):
            start_idx = n_features + j * n_features
            X_new[i, start_idx:start_idx + n_features] = X[i + seq_len - 1 - j]
        for j in range(seq_len):
            X_new[i, -seq_len + j] = y[i + seq_len - 1 - j]
    
    y_new = y[seq_len:]
    return X_new, y_new


def train_sklearn_model(model_name: str, X_train, y_train, X_test, y_test, processor, config, logger):
    """Train sklearn models."""
    seq_len = config.get("seq_len", 24)
    
    # Create features
    X_train_ml, y_train_ml = create_features_for_ml(X_train, y_train, seq_len)
    X_test_ml, y_test_ml = create_features_for_ml(X_test, y_test, seq_len)
    
    # Align test data
    min_len = min(len(X_test_ml), len(y_test_ml))
    X_test_ml = X_test_ml[:min_len]
    y_test_ml = y_test_ml[:min_len]
    
    logger.info(f"    Features: {X_train_ml.shape[1]}")
    
    # Create model
    if model_name == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.1),
            subsample=0.8,
            random_state=42,
            verbose=0
        )
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
    else:
        model = Ridge(alpha=1.0)
    
    # Train
    start_time = time.time()
    model.fit(X_train_ml, y_train_ml)
    train_time = time.time() - start_time
    
    # Predict
    predictions = model.predict(X_test_ml)
    
    # Inverse transform
    predictions_original = processor.inverse_transform_target(predictions)
    y_test_original = processor.inverse_transform_target(y_test_ml)
    
    # Feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        for i, imp in enumerate(model.feature_importances_[:10]):
            feature_importance[f"feature_{i}"] = imp
    
    return {
        "predictions": predictions_original,
        "actual": y_test_original,
        "train_time": train_time,
        "feature_importance": feature_importance
    }


def train_pytorch_model(model_name: str, X_train, y_train, X_test, y_test, processor, config, logger):
    """Train PyTorch models (LSTM, GRU, Transformer)."""
    from src.models.lstm import LSTMModel
    from src.models.gru import GRUForecaster
    from src.models.transformer import TransformerForecaster
    
    seq_len = config.get("seq_len", 24)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"    Device: {device}")
    
    # Create sequences
    X_train_seq, y_train_seq = processor.create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = processor.create_sequences(X_test, y_test, seq_len)
    
    logger.info(f"    Sequence shape: {X_train_seq.shape}")
    
    # Common config
    hidden_size = config.get("hidden_size", 64)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.1)
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 32)
    learning_rate = config.get("learning_rate", 1e-3)
    
    # Create model
    if model_name == "lstm":
        model = LSTMModel(
            seq_len=seq_len,
            pred_len=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        # LSTM needs different input format
        y_train_series = pd.Series(y_train)
        
        start_time = time.time()
        model.fit(y_train_series, verbose=0, early_stopping_patience=10)
        train_time = time.time() - start_time
        
        # Predict
        last_values = y_train[-seq_len:]
        predictions = model.predict(steps=len(y_test), last_values=last_values.reshape(-1, 1))
        y_test_actual = y_test
        
    elif model_name == "gru":
        model = GRUForecaster(
            seq_len=seq_len,
            pred_len=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        
        start_time = time.time()
        model.fit(y_train, verbose=0, early_stopping_patience=10)
        train_time = time.time() - start_time
        
        # Predict
        last_values = y_train[-seq_len:]
        predictions = model.predict(steps=len(y_test), last_values=last_values.reshape(-1, 1))
        y_test_actual = y_test
        
    elif model_name == "transformer":
        model = TransformerForecaster(
            seq_len=seq_len,
            pred_len=1,
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=config.get("dim_feedforward", 256),
            dropout=dropout,
            learning_rate=config.get("learning_rate", 1e-4),
            epochs=epochs,
            batch_size=batch_size
        )
        
        start_time = time.time()
        model.fit(X_train_seq, y_train_seq, verbose=0, early_stopping_patience=15)
        train_time = time.time() - start_time
        
        # Predict
        predictions = model.predict(X_test_seq).flatten()
        y_test_actual = y_test_seq.flatten()
    
    # Inverse transform
    predictions_original = processor.inverse_transform_target(predictions)
    y_test_original = processor.inverse_transform_target(y_test_actual)
    
    return {
        "predictions": predictions_original,
        "actual": y_test_original,
        "train_time": train_time,
        "feature_importance": {}
    }


def run_comparison(
    models: list,
    config_path: str = "configs/wind_power.yaml",
    seq_len: int = 24
):
    """Run model comparison experiment."""
    
    # Load config
    config = load_config(config_path)
    
    # Load and preprocess data
    print("=" * 60)
    print("Wind Power Forecasting - Model Comparison")
    print("=" * 60)
    
    processor = WindPowerDataProcessor(
        target_col="Power",
        time_col="Time",
        scaler_type="minmax"
    )
    
    df = processor.load_data(config["data"]["filepath"])
    print(f"\n[Data] Shape: {df.shape}, Time: {df['Time'].min()} ~ {df['Time'].max()}")
    
    X, y = processor.fit_transform(df, add_time_features=True, add_wind_features=True)
    print(f"[Data] Features: {X.shape[1]}, Samples: {len(X)}")
    
    # Split
    test_size = 0.2
    n_test = int(len(X) * test_size)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    print(f"[Data] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Results storage
    results = {}
    
    # Train each model
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"[{model_name.upper()}] Training...")
        print("=" * 60)
        
        # Setup experiment logging
        model_config = {"seq_len": seq_len, "epochs": 100, "batch_size": 32}
        logger = setup_experiment(model_name, model_config, log_dir="logs")
        
        try:
            if model_name in ["gradient_boosting", "random_forest", "ridge"]:
                result = train_sklearn_model(
                    model_name, X_train, y_train, X_test, y_test,
                    processor, model_config, logger
                )
            else:
                result = train_pytorch_model(
                    model_name, X_train, y_train, X_test, y_test,
                    processor, model_config, logger
                )
            
            # Evaluate
            metrics = evaluate_predictions(result["actual"], result["predictions"])
            
            # Log results
            logger.log_results(metrics)
            logger.info(f"    Training time: {result['train_time']:.2f}s")
            
            # Finalize
            final_metrics = logger.finalize()
            
            results[model_name] = {
                "metrics": metrics,
                "train_time": result["train_time"],
                "status": "success"
            }
            
            print(f"[{model_name.upper()}] R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, Time: {result['train_time']:.2f}s")
            
        except Exception as e:
            print(f"[{model_name.upper()}] FAILED: {str(e)}")
            results[model_name] = {
                "metrics": {},
                "train_time": 0,
                "status": f"failed: {str(e)}"
            }
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, result in results.items():
        if result["status"] == "success":
            m = result["metrics"]
            comparison_data.append({
                "Model": model_name.upper(),
                "R2": f"{m['r2']:.4f}",
                "RMSE": f"{m['rmse']:.4f}",
                "MAE": f"{m['mae']:.4f}",
                "SMAPE": f"{m['smape']:.2f}%",
                "Time(s)": f"{result['train_time']:.1f}"
            })
        else:
            comparison_data.append({
                "Model": model_name.upper(),
                "R2": "FAILED",
                "RMSE": "-",
                "MAE": "-",
                "SMAPE": "-",
                "Time(s)": "-"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save comparison results
    output_dir = Path("logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    comparison_file = output_dir / f"comparison_{timestamp}.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save as CSV
    comparison_csv = output_dir / f"comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    
    print(f"\nComparison saved to: {comparison_file}")
    print(f"CSV saved to: {comparison_csv}")
    
    # Find best model
    best_model = None
    best_r2 = -float('inf')
    for model_name, result in results.items():
        if result["status"] == "success":
            if result["metrics"]["r2"] > best_r2:
                best_r2 = result["metrics"]["r2"]
                best_model = model_name
    
    print(f"\n[BEST MODEL] {best_model.upper()} with R2 = {best_r2:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Model Comparison for Wind Power Forecasting")
    parser.add_argument("--models", type=str, default="gradient_boosting,random_forest,lstm,gru,transformer",
                        help="Comma-separated list of models to compare")
    parser.add_argument("--config", type=str, default="configs/wind_power.yaml", help="Config file path")
    parser.add_argument("--seq-len", type=int, default=24, help="Sequence length")
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(",")]
    
    results = run_comparison(
        models=models,
        config_path=args.config,
        seq_len=args.seq_len
    )


if __name__ == "__main__":
    main()
