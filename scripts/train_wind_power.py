"""
Wind Power Forecasting Training Script
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

from src.models.lstm import LSTMModel
from src.models.transformer import TransformerForecaster
from src.utils.preprocessing import WindPowerDataProcessor
from src.utils.metrics import evaluate_predictions
from src.utils.logging_utils import setup_experiment


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    processor: WindPowerDataProcessor,
    config: dict,
    logger
):
    """Train model and return predictions."""
    model_name = config["model"]["name"]
    model_config = config["model"].get(model_name, {})
    
    if model_name == "lstm":
        seq_len = model_config.get("seq_len", 24)
        
        model = LSTMModel(
            seq_len=seq_len,
            pred_len=1,
            hidden_size=model_config.get("hidden_size", 64),
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.1),
            learning_rate=model_config.get("learning_rate", 1e-3),
            epochs=model_config.get("epochs", 100),
            batch_size=model_config.get("batch_size", 32)
        )
        
        logger.log_training_start("LSTM")
        
        # Convert to pandas Series for LSTM model
        y_train_series = pd.Series(y_train)
        model.fit(y_train_series, verbose=0, early_stopping_patience=10)
        
        # Predict
        last_values = y_train[-seq_len:]
        predictions = model.predict(steps=len(y_test), last_values=last_values.reshape(-1, 1))
        
    elif model_name == "transformer":
        seq_len = model_config.get("seq_len", 24)
        X_train_seq, y_train_seq = processor.create_sequences(X_train, y_train, seq_len)
        X_test_seq, y_test_seq = processor.create_sequences(X_test, y_test, seq_len)
        
        logger.info(f"    Sequence shape: {X_train_seq.shape}")
        
        model = TransformerForecaster(
            seq_len=seq_len,
            pred_len=1,
            d_model=model_config.get("d_model", 64),
            nhead=model_config.get("nhead", 4),
            num_encoder_layers=model_config.get("num_encoder_layers", 2),
            num_decoder_layers=model_config.get("num_decoder_layers", 2),
            dim_feedforward=model_config.get("dim_feedforward", 256),
            dropout=model_config.get("dropout", 0.1),
            learning_rate=model_config.get("learning_rate", 1e-4),
            epochs=model_config.get("epochs", 100),
            batch_size=model_config.get("batch_size", 32)
        )
        
        logger.log_training_start("Transformer")
        
        train_start = datetime.now()
        model.fit(X_train_seq, y_train_seq, verbose=0, early_stopping_patience=15)
        train_duration = (datetime.now() - train_start).total_seconds()
        logger.info(f"    Training time: {train_duration:.2f} seconds")
        
        predictions = model.predict(X_test_seq).flatten()
        y_test = y_test_seq.flatten()
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, predictions, y_test


def main():
    parser = argparse.ArgumentParser(description="Wind Power Forecasting")
    parser.add_argument("--config", type=str, default="configs/wind_power.yaml", help="Config file path")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], help="Override model type")
    parser.add_argument("--seq-len", type=int, help="Override sequence length")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.model:
        config["model"]["name"] = args.model
    if args.seq_len:
        config["model"][config["model"]["name"]]["seq_len"] = args.seq_len
    if args.epochs:
        config["model"][config["model"]["name"]]["epochs"] = args.epochs
    if args.batch_size:
        config["model"][config["model"]["name"]]["batch_size"] = args.batch_size
    
    model_name = config["model"]["name"]
    model_config = config["model"].get(model_name, {})
    
    # Setup logging
    logger = setup_experiment(model_name, model_config, log_dir="logs")
    
    # Load data
    logger.info("\n[1/4] Loading data...")
    processor = WindPowerDataProcessor(
        target_col="Power",
        time_col="Time",
        scaler_type="minmax"
    )
    
    df = processor.load_data(config["data"]["filepath"])
    time_range = f"{df['Time'].min()} ~ {df['Time'].max()}"
    logger.info(f"    Data shape: {df.shape}")
    logger.info(f"    Time range: {time_range}")
    
    # Preprocess
    logger.info("\n[2/4] Preprocessing...")
    X, y = processor.fit_transform(df, add_time_features=True, add_wind_features=True)
    logger.info(f"    Features: {X.shape[1]}")
    logger.info(f"    Samples: {len(X)}")
    
    # Split
    X_train, X_test, y_train, y_test = processor.train_test_split(
        X, y, test_size=config["data"]["test_size"], shuffle=False
    )
    
    logger.log_data_info(
        n_samples=len(X),
        n_features=X.shape[1],
        n_train=len(X_train),
        n_test=len(X_test),
        time_range=time_range
    )
    
    # Train
    logger.info("\n[3/4] Training model...")
    model, predictions, y_test_actual = train_model(
        X_train, y_train, X_test, y_test, processor, config, logger
    )
    
    # Inverse transform predictions
    predictions_original = processor.inverse_transform_target(predictions)
    y_test_original = processor.inverse_transform_target(y_test_actual)
    
    # Evaluate
    logger.info("\n[4/4] Evaluating...")
    metrics = evaluate_predictions(y_test_original, predictions_original)
    logger.log_results(metrics)
    
    # Save results
    output_path = Path(config["output"]["results_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    pred_file = output_path / f"predictions_{model_name}_{timestamp}.csv"
    pred_df = pd.DataFrame({
        "actual": y_test_original,
        "predicted": predictions_original,
        "error": y_test_original - predictions_original
    })
    pred_df.to_csv(pred_file, index=False)
    logger.log_file_saved("Predictions", str(pred_file))
    
    # Save model
    model_path = Path(config["output"]["model_path"])
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{model_name}_model.pkl"
    processor_file = model_path / "processor.pkl"
    model.save(str(model_file))
    processor.save(str(processor_file))
    logger.log_file_saved("Model", str(model_file))
    
    # Finalize logging
    final_metrics = logger.finalize()
    
    logger.log_separator()
    logger.info("✅ Training completed!")
    logger.log_separator()


if __name__ == "__main__":
    main()
