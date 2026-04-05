"""
Train All Baseline Models and Compare Results

Run all models: LSTM, GRU, Transformer, CNN-GRU, GradientBoosting, RandomForest
Compare metrics and generate leaderboard.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.lstm import LSTMModel
from src.models.gru import GRUForecaster
from src.models.transformer import TransformerForecaster
from src.models.cnn_gru import CNNGRUForecaster
from src.utils.metrics import evaluate_predictions


def load_and_prepare_data(data_path: str, seq_len: int = 24) -> dict:
    """Load and prepare data for all models."""
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(data_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    # Features
    feature_cols = [col for col in df.columns if col not in ['Time', 'Power']]
    target_col = 'Power'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split
    test_size = 0.2
    split_idx = int(len(X_scaled) * (1 - test_size))
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # Create sequences for sequence models
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)
    
    return {
        'X_train_seq': X_train_seq,
        'y_train_seq': y_train_seq,
        'X_test_seq': X_test_seq,
        'y_test_seq': y_test_seq,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'n_features': X.shape[1],
        'seq_len': seq_len,
        'feature_cols': feature_cols
    }


def train_gradient_boosting(data: dict, verbose: int = 1) -> dict:
    """Train Gradient Boosting model."""
    print("\n" + "="*70)
    print("TRAINING GRADIENT BOOSTING")
    print("="*70)
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Flatten sequences
    X_train_flat = data['X_train_seq'].reshape(data['X_train_seq'].shape[0], -1)
    X_test_flat = data['X_test_seq'].reshape(data['X_test_seq'].shape[0], -1)
    y_train_flat = data['y_train_seq'].flatten()
    y_test_flat = data['y_test_seq'].flatten()
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train_flat, y_train_flat)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_flat)
    y_true = y_test_flat
    
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"\nResults:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  Time: {train_time:.1f}s")
    
    return {'model': 'GradientBoosting', 'metrics': metrics}


def train_random_forest(data: dict, verbose: int = 1) -> dict:
    """Train Random Forest model."""
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)
    
    from sklearn.ensemble import RandomForestRegressor
    
    # Flatten sequences
    X_train_flat = data['X_train_seq'].reshape(data['X_train_seq'].shape[0], -1)
    X_test_flat = data['X_test_seq'].reshape(data['X_test_seq'].shape[0], -1)
    y_train_flat = data['y_train_seq'].flatten()
    y_test_flat = data['y_test_seq'].flatten()
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    model.fit(X_train_flat, y_train_flat)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_flat)
    y_true = y_test_flat
    
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"\nResults:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  Time: {train_time:.1f}s")
    
    return {'model': 'RandomForest', 'metrics': metrics}


def train_xgboost(data: dict, verbose: int = 1) -> dict:
    """Train XGBoost model."""
    print("\n" + "="*70)
    print("TRAINING XGBOOST")
    print("="*70)
    
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed, skipping...")
        return {'model': 'XGBoost', 'metrics': {'r2': 0, 'error': 'not_installed'}}
    
    # Flatten sequences
    X_train_flat = data['X_train_seq'].reshape(data['X_train_seq'].shape[0], -1)
    X_test_flat = data['X_test_seq'].reshape(data['X_test_seq'].shape[0], -1)
    y_train_flat = data['y_train_seq'].flatten()
    y_test_flat = data['y_test_seq'].flatten()
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    start_time = time.time()
    model.fit(X_train_flat, y_train_flat)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_flat)
    y_true = y_test_flat
    
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"\nResults:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  Time: {train_time:.1f}s")
    
    return {'model': 'XGBoost', 'metrics': metrics}


def train_lstm(data: dict, verbose: int = 1) -> dict:
    """Train LSTM model."""
    print("\n" + "="*70)
    print("TRAINING LSTM")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = LSTMModel(
        seq_len=data['seq_len'],
        pred_len=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        device=device
    )
    
    # LSTM needs pd.Series format
    y_train_series = pd.Series(data['y_train'])
    
    start_time = time.time()
    model.fit(y_train_series, verbose=verbose)
    train_time = time.time() - start_time
    
    # Predict - need last values from training
    last_values = data['y_train'][-data['seq_len']:]
    y_pred = model.predict(steps=len(data['y_test_seq']), last_values=last_values)
    
    # Get corresponding true values
    y_true = data['y_test'][data['seq_len']:data['seq_len']+len(y_pred)]
    
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"\nResults:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  Time: {train_time:.1f}s")
    
    return {'model': 'LSTM', 'metrics': metrics}


def train_gru(data: dict, verbose: int = 1) -> dict:
    """Train GRU model."""
    print("\n" + "="*70)
    print("TRAINING GRU")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GRUForecaster(
        seq_len=data['seq_len'],
        pred_len=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        device=device
    )
    
    start_time = time.time()
    model.fit(data['y_train'], verbose=verbose)
    train_time = time.time() - start_time
    
    # Predict using last values
    last_values = data['y_train'][-data['seq_len']:]
    y_pred = model.predict(steps=len(data['y_test_seq']), last_values=last_values)
    
    y_true = data['y_test'][data['seq_len']:data['seq_len']+len(y_pred)]
    
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"\nResults:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  Time: {train_time:.1f}s")
    
    return {'model': 'GRU', 'metrics': metrics}


def train_cnn_gru(data: dict, verbose: int = 1) -> dict:
    """Train CNN-GRU model."""
    print("\n" + "="*70)
    print("TRAINING CNN-GRU")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CNNGRUForecaster(
        seq_len=data['seq_len'],
        pred_len=1,
        input_features=data['n_features'],
        cnn_channels=[32, 64],
        cnn_kernel_sizes=[3, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        dropout=0.1,
        use_batch_norm=True,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        device=device
    )
    
    start_time = time.time()
    model.fit(data['X_train_seq'], data['y_train_seq'].flatten(), verbose=verbose)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X=data['X_test_seq'])
    y_true = data['y_test_seq'].flatten()
    
    metrics = model.evaluate(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"\nResults:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  Time: {train_time:.1f}s")
    
    return {'model': 'CNN-GRU', 'metrics': metrics}


def train_transformer(data: dict, verbose: int = 1) -> dict:
    """Train Transformer model."""
    print("\n" + "="*70)
    print("TRAINING TRANSFORMER")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TransformerForecaster(
        seq_len=data['seq_len'],
        pred_len=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dim_feedforward=256,
        dropout=0.1,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        device=device
    )
    
    start_time = time.time()
    model.fit(data['X_train_seq'], data['y_train_seq'], verbose=verbose)
    train_time = time.time() - start_time
    
    y_pred = model.predict(data['X_test_seq']).flatten()
    y_true = data['y_test_seq'].flatten()
    
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"\nResults:")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  Time: {train_time:.1f}s")
    
    return {'model': 'Transformer', 'metrics': metrics}


def run_all_models(
    data_path: str,
    seq_len: int = 24,
    verbose: int = 1,
    output_dir: str = "data/results"
) -> list:
    """Run all models and return results."""
    
    print("\n" + "#"*70)
    print("# MODEL COMPARISON TRAINING")
    print("#"*70)
    print(f"Data: {data_path}")
    print(f"Seq Length: {seq_len}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Prepare data
    print("\nPreparing data...")
    data = load_and_prepare_data(data_path, seq_len)
    print(f"Train sequences: {data['X_train_seq'].shape}")
    print(f"Test sequences: {data['X_test_seq'].shape}")
    print(f"Features: {data['n_features']}")
    
    results = []
    
    # Train all models (order by expected speed)
    models_to_train = [
        ("GradientBoosting", train_gradient_boosting),
        ("RandomForest", train_random_forest),
        ("XGBoost", train_xgboost),
        ("LSTM", train_lstm),
        ("GRU", train_gru),
        ("CNN-GRU", train_cnn_gru),
        ("Transformer", train_transformer),
    ]
    
    for name, train_func in models_to_train:
        try:
            print(f"\n{'='*70}")
            result = train_func(data, verbose)
            if 'error' not in result['metrics']:
                results.append(result)
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort by R²
    results = sorted(results, key=lambda x: x['metrics']['r2'], reverse=True)
    
    # Print leaderboard
    print("\n" + "="*70)
    print("MODEL LEADERBOARD (sorted by R²)")
    print("="*70)
    print(f"{'Rank':<6} {'Model':<20} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'Time':>10}")
    print("-"*70)
    
    for i, r in enumerate(results, 1):
        m = r['metrics']
        status = "✓ TARGET" if m['r2'] >= 0.95 else ""
        print(f"{i:<6} {r['model']:<20} {m['r2']:>10.4f} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m.get('train_time', 0):>10.1f}s {status}")
    
    print("="*70)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"all_models_comparison_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train All Models")
    parser.add_argument("--data", type=str, default="dataset/Location1.csv")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output", type=str, default="data/results")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data
    output_dir = project_root / args.output
    
    results = run_all_models(
        str(data_path),
        args.seq_len,
        args.verbose,
        str(output_dir)
    )
    
    # Summary
    if results:
        best = results[0]
        print(f"\n🏆 BEST MODEL: {best['model']} with R² = {best['metrics']['r2']:.4f}")
        
        target_met = sum(1 for r in results if r['metrics']['r2'] >= 0.95)
        print(f"\n📊 Models meeting target (R² >= 0.95): {target_met}/{len(results)}")


if __name__ == "__main__":
    main()