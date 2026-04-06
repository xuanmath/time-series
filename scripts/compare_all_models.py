"""
Unified Model Comparison - 统一模型对比脚本

解决接口不一致问题，所有模型统一处理
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ============================================================
# Data Preparation
# ============================================================

def prepare_data(data_path, seq_len=24):
    """Prepare data for all models."""
    df = pd.read_csv(data_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col not in ['Time', 'Power']]
    target_col = 'Power'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # Create sequences for deep learning models
    def create_seq(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_seq(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = create_seq(X_test, y_test, seq_len)
    
    return {
        'X_train_seq': X_train_seq, 'y_train_seq': y_train_seq,
        'X_test_seq': X_test_seq, 'y_test_seq': y_test_seq,
        'X_train': X_train, 'y_train': y_train,
        'X_test': y_test, 'y_test': y_test,
        'n_features': len(feature_cols),
        'seq_len': seq_len,
        'feature_cols': feature_cols
    }


def calc_metrics(y_true, y_pred):
    """Calculate metrics."""
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


# ============================================================
# Sklearn Models
# ============================================================

def train_gradient_boosting(data):
    """Gradient Boosting."""
    print("\n" + "="*50)
    print("Training: GradientBoosting")
    print("="*50)
    
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
    metrics = calc_metrics(y_test, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    print(f"Time: {train_time:.1f}s")
    
    return {'model': 'GradientBoosting', 'metrics': metrics}


def train_random_forest(data):
    """Random Forest."""
    print("\n" + "="*50)
    print("Training: RandomForest")
    print("="*50)
    
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
    metrics = calc_metrics(y_test, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    print(f"Time: {train_time:.1f}s")
    
    return {'model': 'RandomForest', 'metrics': metrics}


def train_xgboost(data):
    """XGBoost."""
    print("\n" + "="*50)
    print("Training: XGBoost")
    print("="*50)
    
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed, skipping...")
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
    metrics = calc_metrics(y_test, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    print(f"Time: {train_time:.1f}s")
    
    return {'model': 'XGBoost', 'metrics': metrics}


# ============================================================
# PyTorch Models
# ============================================================

class SimpleLSTM(torch.nn.Module):
    """Simple LSTM for multivariate input."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SimpleGRU(torch.nn.Module):
    """Simple GRU for multivariate input."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class SimpleTransformer(torch.nn.Module):
    """Simple Transformer encoder for time series."""
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])


class SimpleCNNGRU(torch.nn.Module):
    """CNN + GRU hybrid."""
    def __init__(self, input_size, hidden_size=64, dropout=0.1):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(64, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def train_pytorch_model(data, model_class, model_name, epochs=30, batch_size=64, lr=0.001):
    """Generic PyTorch model training."""
    print("\n" + "="*50)
    print(f"Training: {model_name}")
    print("="*50)
    
    X_train = torch.FloatTensor(data['X_train_seq']).to(DEVICE)
    y_train = torch.FloatTensor(data['y_train_seq']).to(DEVICE)
    X_test = torch.FloatTensor(data['X_test_seq']).to(DEVICE)
    y_test = data['y_test_seq'].flatten()
    
    # Create model
    model = model_class(input_size=data['n_features']).to(DEVICE)
    
    # Training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    t0 = time.time()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    train_time = time.time() - t0
    
    # Predict
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()
    
    metrics = calc_metrics(y_test, y_pred)
    metrics['train_time'] = train_time
    
    print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    print(f"Time: {train_time:.1f}s")
    
    return {'model': model_name, 'metrics': metrics}


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "#"*50)
    print("# MODEL COMPARISON - ALL MODELS")
    print("#"*50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data_path = project_root / "dataset/Location1.csv"
    print(f"\nLoading data from: {data_path}")
    
    data = prepare_data(str(data_path), seq_len=24)
    print(f"Train: {data['X_train_seq'].shape}")
    print(f"Test: {data['X_test_seq'].shape}")
    print(f"Features: {data['n_features']}")
    
    results = []
    
    # 1. Sklearn models (fast)
    print("\n" + "-"*50)
    print("SKLEARN MODELS")
    print("-"*50)
    
    try:
        r = train_gradient_boosting(data)
        if r: results.append(r)
    except Exception as e:
        print(f"GradientBoosting failed: {e}")
    
    try:
        r = train_random_forest(data)
        if r: results.append(r)
    except Exception as e:
        print(f"RandomForest failed: {e}")
    
    try:
        r = train_xgboost(data)
        if r: results.append(r)
    except Exception as e:
        print(f"XGBoost failed: {e}")
    
    # 2. PyTorch models
    print("\n" + "-"*50)
    print("PYTORCH MODELS")
    print("-"*50)
    
    pytorch_models = [
        (SimpleLSTM, "LSTM"),
        (SimpleGRU, "GRU"),
        (SimpleTransformer, "Transformer"),
        (SimpleCNNGRU, "CNN-GRU"),
    ]
    
    for model_class, name in pytorch_models:
        try:
            r = train_pytorch_model(data, model_class, name, epochs=30)
            if r: results.append(r)
        except Exception as e:
            print(f"{name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort and display
    results = sorted(results, key=lambda x: x['metrics']['r2'], reverse=True)
    
    print("\n" + "="*50)
    print("LEADERBOARD (sorted by R²)")
    print("="*50)
    print(f"{'Rank':<6}{'Model':<18}{'R²':>10}{'RMSE':>10}{'MAE':>10}{'Time':>8}")
    print("-"*50)
    
    for i, r in enumerate(results, 1):
        m = r['metrics']
        status = "✓" if m['r2'] >= 0.95 else ""
        print(f"{i:<6}{r['model']:<18}{m['r2']:>10.4f}{m['rmse']:>10.4f}{m['mae']:>10.4f}{m['train_time']:>7.1f}s {status}")
    
    print("="*50)
    
    # Save
    output_dir = project_root / "data/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"model_comparison_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved: {output_file}")
    
    if results:
        best = results[0]
        print(f"\n🏆 BEST: {best['model']} with R² = {best['metrics']['r2']:.4f}")
        
        target_met = sum(1 for r in results if r['metrics']['r2'] >= 0.95)
        print(f"📊 Models with R² >= 0.95: {target_met}/{len(results)}")
    
    return results


if __name__ == "__main__":
    main()