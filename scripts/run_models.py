"""
Train models step by step - simpler version for testing
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.metrics import evaluate_predictions
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def prepare_data(data_path, seq_len=24):
    """Load and prepare data."""
    df = pd.read_csv(data_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col not in ['Time', 'Power']]
    X = df[feature_cols].values
    y = df['Power'].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # Sequences
    X_train_seq, y_train_seq = [], []
    for i in range(len(X_train) - seq_len):
        X_train_seq.append(X_train[i:i+seq_len])
        y_train_seq.append(y_train[i+seq_len])
    
    X_test_seq, y_test_seq = [], []
    for i in range(len(X_test) - seq_len):
        X_test_seq.append(X_test[i:i+seq_len])
        y_test_seq.append(y_test[i+seq_len])
    
    return {
        'X_train_seq': np.array(X_train_seq),
        'y_train_seq': np.array(y_train_seq),
        'X_test_seq': np.array(X_test_seq),
        'y_test_seq': np.array(y_test_seq),
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'n_features': len(feature_cols),
        'seq_len': seq_len,
        'feature_cols': feature_cols
    }


def train_gradient_boosting(data):
    """Train GB model."""
    print("\n" + "="*60)
    print("GRADIENT BOOSTING")
    print("="*60)
    
    X_train = data['X_train_seq'].reshape(len(data['X_train_seq']), -1)
    X_test = data['X_test_seq'].reshape(len(data['X_test_seq']), -1)
    y_train = data['y_train_seq'].flatten()
    y_test = data['y_test_seq'].flatten()
    
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    return {'model': 'GradientBoosting', 'metrics': metrics}


def train_random_forest(data):
    """Train RF model."""
    print("\n" + "="*60)
    print("RANDOM FOREST")
    print("="*60)
    
    X_train = data['X_train_seq'].reshape(len(data['X_train_seq']), -1)
    X_test = data['X_test_seq'].reshape(len(data['X_test_seq']), -1)
    y_train = data['y_train_seq'].flatten()
    y_test = data['y_test_seq'].flatten()
    
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    return {'model': 'RandomForest', 'metrics': metrics}


def train_xgboost(data):
    """Train XGBoost."""
    print("\n" + "="*60)
    print("XGBOOST")
    print("="*60)
    
    try:
        import xgboost as xgb
    except:
        print("XGBoost not installed")
        return None
    
    X_train = data['X_train_seq'].reshape(len(data['X_train_seq']), -1)
    X_test = data['X_test_seq'].reshape(len(data['X_test_seq']), -1)
    y_train = data['y_train_seq'].flatten()
    y_test = data['y_test_seq'].flatten()
    
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42, n_jobs=-1, verbosity=0
    )
    
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    return {'model': 'XGBoost', 'metrics': metrics}


def train_lstm(data):
    """Train LSTM."""
    print("\n" + "="*60)
    print("LSTM")
    print("="*60)
    
    from src.models.lstm import LSTMModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = LSTMModel(
        seq_len=data['seq_len'],
        hidden_size=64,
        num_layers=2,
        epochs=30,
        batch_size=64,
        device=device
    )
    
    t0 = time.time()
    model.fit(pd.Series(data['y_train']), verbose=1)
    train_time = time.time() - t0
    
    last_vals = data['y_train'][-data['seq_len']:]
    y_pred = model.predict(steps=len(data['y_test_seq']), last_values=last_vals)
    y_true = data['y_test'][data['seq_len']:data['seq_len']+len(y_pred)]
    
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    return {'model': 'LSTM', 'metrics': metrics}


def train_gru(data):
    """Train GRU."""
    print("\n" + "="*60)
    print("GRU")
    print("="*60)
    
    from src.models.gru import GRUForecaster
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GRUForecaster(
        seq_len=data['seq_len'],
        hidden_size=64,
        num_layers=2,
        epochs=30,
        batch_size=64,
        device=device
    )
    
    t0 = time.time()
    model.fit(data['y_train'], verbose=1)
    train_time = time.time() - t0
    
    last_vals = data['y_train'][-data['seq_len']:]
    y_pred = model.predict(steps=len(data['y_test_seq']), last_values=last_vals)
    y_true = data['y_test'][data['seq_len']:data['seq_len']+len(y_pred)]
    
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    return {'model': 'GRU', 'metrics': metrics}


def train_cnn_gru(data):
    """Train CNN-GRU."""
    print("\n" + "="*60)
    print("CNN-GRU")
    print("="*60)
    
    from src.models.cnn_gru import CNNGRUForecaster
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CNNGRUForecaster(
        seq_len=data['seq_len'],
        input_features=data['n_features'],
        cnn_channels=[32, 64],
        cnn_kernel_sizes=[3, 3],
        gru_hidden_size=64,
        gru_num_layers=2,
        epochs=30,
        batch_size=64,
        device=device
    )
    
    t0 = time.time()
    model.fit(data['X_train_seq'], data['y_train_seq'].flatten(), verbose=1)
    train_time = time.time() - t0
    
    y_pred = model.predict(X=data['X_test_seq'])
    y_true = data['y_test_seq'].flatten()
    
    metrics = model.evaluate(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    return {'model': 'CNN-GRU', 'metrics': metrics}


def train_transformer(data):
    """Train Transformer."""
    print("\n" + "="*60)
    print("TRANSFORMER")
    print("="*60)
    
    from src.models.transformer import TransformerForecaster
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TransformerForecaster(
        seq_len=data['seq_len'],
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dim_feedforward=256,
        epochs=30,
        batch_size=64,
        device=device
    )
    
    t0 = time.time()
    model.fit(data['X_train_seq'], data['y_train_seq'], verbose=1)
    train_time = time.time() - t0
    
    y_pred = model.predict(data['X_test_seq']).flatten()
    y_true = data['y_test_seq'].flatten()
    
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    return {'model': 'Transformer', 'metrics': metrics}


def main():
    print("\n" + "#"*60)
    print("# MODEL COMPARISON TRAINING")
    print("#"*60)
    
    data_path = project_root / "dataset/Location1.csv"
    print(f"Data: {data_path}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Prepare data
    print("\nPreparing data...")
    data = prepare_data(str(data_path), seq_len=24)
    print(f"Train: {data['X_train_seq'].shape}, Test: {data['X_test_seq'].shape}")
    print(f"Features: {data['n_features']}")
    
    results = []
    
    # Train each model
    models = [
        ("GradientBoosting", train_gradient_boosting),
        ("RandomForest", train_random_forest),
        ("XGBoost", train_xgboost),
        ("GRU", train_gru),
        ("CNN-GRU", train_cnn_gru),
        ("Transformer", train_transformer),
    ]
    
    for name, func in models:
        try:
            result = func(data)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Sort and print
    results = sorted(results, key=lambda x: x['metrics']['r2'], reverse=True)
    
    print("\n" + "="*60)
    print("LEADERBOARD")
    print("="*60)
    print(f"{'Rank':<6}{'Model':<20}{'R²':>10}{'RMSE':>10}{'MAE':>10}")
    print("-"*60)
    
    for i, r in enumerate(results, 1):
        m = r['metrics']
        status = "✓" if m['r2'] >= 0.95 else ""
        print(f"{i:<6}{r['model']:<20}{m['r2']:>10.4f}{m['rmse']:>10.4f}{m['mae']:>10.4f} {status}")
    
    # Save
    output = project_root / "data/results"
    output.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output / f"comparison_{ts}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved: {output / f'comparison_{ts}.json'}")
    
    if results:
        best = results[0]
        print(f"\n🏆 BEST: {best['model']} R²={best['metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()