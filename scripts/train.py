"""
Training script for time series models
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ARIMAModel, ProphetModel, LSTMModel
from src.features import FeatureEngineer
from src.utils import load_data, evaluate_predictions, plot_forecast
from src.utils.data import generate_sample_data


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_arima(config: dict, y_train: pd.Series, y_test: pd.Series):
    """Train ARIMA model."""
    model_config = config["model"]["arima"]
    
    model = ARIMAModel(
        order=tuple(model_config["order"]),
        seasonal_order=tuple(model_config["seasonal_order"]) if model_config["seasonal_order"][0] > 0 else None
    )
    
    model.fit(y_train)
    
    # Predict
    predictions = model.predict(steps=len(y_test))
    
    return model, predictions


def train_prophet(config: dict, df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Train Prophet model."""
    model_config = config["model"]["prophet"]
    
    # Prepare data for Prophet
    prophet_df = df_train[["date", "value"]].copy()
    prophet_df.columns = ["ds", "y"]
    
    model = ProphetModel(
        growth=model_config["growth"],
        seasonality_mode=model_config["seasonality_mode"],
        yearly_seasonality=model_config["yearly_seasonality"],
        weekly_seasonality=model_config["weekly_seasonality"],
        daily_seasonality=model_config["daily_seasonality"]
    )
    
    model.fit(prophet_df)
    
    # Predict
    forecast = model.predict(periods=len(df_test))
    predictions = forecast.iloc[-len(df_test):]["yhat"].values
    
    return model, predictions


def train_lstm(config: dict, y_train: pd.Series, y_test: pd.Series):
    """Train LSTM model."""
    model_config = config["model"]["lstm"]
    
    model = LSTMModel(
        seq_len=model_config["seq_len"],
        pred_len=model_config["pred_len"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        learning_rate=model_config["learning_rate"],
        epochs=model_config["epochs"],
        batch_size=model_config["batch_size"]
    )
    
    model.fit(y_train, verbose=1)
    
    # Predict
    last_values = y_train.values[-model.seq_len:]
    predictions = model.predict(steps=len(y_test), last_values=last_values)
    
    return model, predictions


def main():
    parser = argparse.ArgumentParser(description="Train time series model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, choices=["arima", "prophet", "lstm"], help="Override model type")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.model:
        config["model"]["name"] = args.model
    
    print(f"Training {config['model']['name']} model...")
    
    # Load or generate data
    data_config = config["data"]
    
    if Path(data_config["filepath"]).exists():
        df = load_data(
            data_config["filepath"],
            time_col=data_config["time_col"],
            value_col=data_config["value_col"]
        )
        df = df.reset_index()
    else:
        print("Generating sample data...")
        df = generate_sample_data(n_samples=500)
    
    # Split data
    test_size = int(len(df) * data_config["test_size"])
    df_train = df.iloc[:-test_size]
    df_test = df.iloc[-test_size:]
    
    y_train = df_train.set_index("date")["value"]
    y_test = df_test.set_index("date")["value"]
    
    # Train model
    model_name = config["model"]["name"]
    
    if model_name == "arima":
        model, predictions = train_arima(config, y_train, y_test)
    elif model_name == "prophet":
        model, predictions = train_prophet(config, df_train, df_test)
    elif model_name == "lstm":
        model, predictions = train_lstm(config, y_train, y_test)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Evaluate
    metrics = evaluate_predictions(y_test.values, predictions)
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:.4f}")
    
    # Save results
    output_config = config["output"]
    Path(output_config["results_path"]).mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    if output_config["save_predictions"]:
        pred_df = pd.DataFrame({
            "date": df_test["date"],
            "actual": y_test.values,
            "predicted": predictions
        })
        pred_df.to_csv(f"{output_config['results_path']}/predictions.csv", index=False)
        print(f"\nPredictions saved to {output_config['results_path']}/predictions.csv")
    
    # Save model
    if output_config["model_path"]:
        Path(output_config["model_path"]).mkdir(parents=True, exist_ok=True)
        model.save(f"{output_config['model_path']}/{model_name}_model.pkl")
        print(f"Model saved to {output_config['model_path']}/{model_name}_model.pkl")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
