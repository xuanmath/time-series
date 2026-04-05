"""
Train CNN-GRU Model with Location1.csv Dataset

This script trains a CNN-GRU model for time series forecasting
using the Location1.csv wind power data.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.cnn_gru import CNNGRUForecaster, get_default_config, get_advanced_config
import torch


def load_location1_data(filepath: str) -> pd.DataFrame:
    """Load Location1.csv dataset."""
    df = pd.read_csv(filepath)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame, target_col: str = 'Power') -> tuple:
    """
    Prepare features and target for CNN-GRU.
    
    Features: temperature, humidity, windspeed, winddirection, etc.
    Target: Power
    """
    # Feature columns (excluding Time and Power)
    feature_cols = [col for col in df.columns if col not in ['Time', target_col]]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y, feature_cols


def train_and_evaluate(
    data_path: str,
    config: dict,
    output_dir: str,
    verbose: int = 1
) -> dict:
    """
    Train CNN-GRU model and evaluate performance.
    
    Returns metrics dict with R2, RMSE, MAE, SMAPE.
    """
    print("=" * 70)
    print("CNN-GRU MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    df = load_location1_data(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    print(f"Features: {feature_cols}")
    print(f"Target: Power")
    
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split data
    test_size = 0.2
    split_idx = int(len(X_scaled) * (1 - test_size))
    
    X_train = X_scaled[:split_idx]
    y_train = y_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_test = y_scaled[split_idx:]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    model = CNNGRUForecaster(
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        input_features=len(feature_cols),
        cnn_channels=config['cnn_channels'],
        cnn_kernel_sizes=config['cnn_kernel_sizes'],
        gru_hidden_size=config['gru_hidden_size'],
        gru_num_layers=config['gru_num_layers'],
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm'],
        learning_rate=config['learning_rate'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        weight_decay=config['weight_decay'],
        device=device
    )
    
    # Train model
    print("\n" + "-" * 70)
    print("Training...")
    print("-" * 70)
    
    train_history = model.fit(
        X_train, 
        y_train,
        verbose=verbose,
        early_stopping_patience=config.get('early_stopping_patience', 15)
    )
    
    # Create test sequences
    X_test_seq = []
    y_test_seq = []
    
    for i in range(len(X_test) - config['seq_len']):
        X_test_seq.append(X_test[i:i + config['seq_len']])
        y_test_seq.append(y_test[i + config['seq_len']])
    
    X_test_seq = np.array(X_test_seq)
    y_test_seq = np.array(y_test_seq)
    
    # Predict
    y_pred_scaled = model.predict(X=X_test_seq)
    
    # Inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    
    # Evaluate
    metrics = model.evaluate(y_true, y_pred)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"R²:    {metrics['r2']:.6f}")
    print(f"RMSE:  {metrics['rmse']:.6f}")
    print(f"MAE:   {metrics['mae']:.6f}")
    print(f"SMAPE: {metrics['smape']:.2f}%")
    print(f"Best epoch: {train_history['best_epoch']}")
    print("=" * 70)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_file = output_path / f"metrics_cnn_gru_{timestamp}.json"
    result_data = {
        "model": "cnn_gru",
        "params": config,
        "metrics": metrics,
        "training": {
            "best_epoch": train_history['best_epoch'],
            "best_val_loss": train_history['best_val_loss'],
            "total_epochs": train_history['total_epochs']
        },
        "data": {
            "filepath": data_path,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": feature_cols
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'error': y_true - y_pred
    })
    predictions_file = output_path / f"predictions_cnn_gru_{timestamp}.csv"
    predictions_df.to_csv(predictions_file, index=False)
    
    # Save model
    model_file = output_path / f"cnn_gru_model_{timestamp}.pt"
    model.save(str(model_file))
    
    print(f"\nResults saved to {output_path}")
    
    return result_data


def main():
    parser = argparse.ArgumentParser(description="Train CNN-GRU Model")
    parser.add_argument("--data", type=str, 
                        default="dataset/Location1.csv",
                        help="Data file path")
    parser.add_argument("--config", type=str, default="default",
                        choices=["default", "advanced"],
                        help="Model configuration")
    parser.add_argument("--output", type=str, default="data/results",
                        help="Output directory")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbose level (0, 1, 2)")
    
    args = parser.parse_args()
    
    # Get config
    if args.config == "advanced":
        config = get_advanced_config()
    else:
        config = get_default_config()
    
    # Resolve data path
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data
    
    output_dir = project_root / args.output
    
    result = train_and_evaluate(
        str(data_path),
        config,
        str(output_dir),
        args.verbose
    )
    
    # Return success/fail based on R2
    if result['metrics']['r2'] >= 0.95:
        print("\n✅ TARGET ACHIEVED: R² >= 0.95")
    else:
        print(f"\n⚠️ Target not met: R² = {result['metrics']['r2']:.4f} (target: 0.95)")
    
    return result


if __name__ == "__main__":
    main()