"""
Deep Learning Model Comparison - Enhanced Version

更长的训练轮数，更好的超参数
"""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def prepare_data(data_path, seq_len=48):
    import pandas as pd
    
    df = pd.read_csv(data_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col not in ['Time', 'Power']]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['Power'].values.astype(np.float32)
    
    # Normalize
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-10
    y_mean, y_std = y.mean(), y.std() + 1e-10
    
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
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
        'n_features': len(feature_cols),
        'seq_len': seq_len
    }


class EnhancedLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=0.2
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class EnhancedGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=0.2
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class EnhancedCNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 64, 3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, 3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, 3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(128, hidden_size, 2, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class EnhancedTransformer(torch.nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        self.proj = torch.nn.Linear(input_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, batch_first=True, dropout=0.2
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.proj(x)
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


def train_model(data, model_class, name, epochs=100, batch_size=64, lr=0.001):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    X_train = torch.FloatTensor(data['X_train_seq']).to(DEVICE)
    y_train = torch.FloatTensor(data['y_train_seq']).to(DEVICE)
    X_test = torch.FloatTensor(data['X_test_seq']).to(DEVICE)
    y_test = data['y_test_seq'].flatten()
    
    model = model_class(input_size=data['n_features']).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    t0 = time.time()
    best_loss = float('inf')
    patience = 0
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step()
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            best_state = model.state_dict().copy()
        else:
            patience += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")
        
        if patience >= 20:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    train_time = time.time() - t0
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': rmse(y_test, y_pred),
        'mae': mae(y_test, y_pred),
        'train_time': train_time
    }
    
    print(f"R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    print(f"Time: {train_time:.1f}s")
    
    return {'model': name, 'metrics': metrics}


def main():
    print("\n" + "#"*60)
    print("# ENHANCED DEEP LEARNING MODEL COMPARISON")
    print("#"*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data_path = project_root / "dataset/Location1.csv"
    print(f"\nLoading: {data_path}")
    
    # Use longer sequence
    data = prepare_data(str(data_path), seq_len=48)
    print(f"Train: {data['X_train_seq'].shape}")
    print(f"Test: {data['X_test_seq'].shape}")
    
    results = []
    
    models = [
        (EnhancedLSTM, "LSTM-Enhanced"),
        (EnhancedGRU, "GRU-Enhanced"),
        (EnhancedCNN, "CNN-GRU-Enhanced"),
        (EnhancedTransformer, "Transformer-Enhanced"),
    ]
    
    for model_class, name in models:
        try:
            r = train_model(data, model_class, name, epochs=100)
            results.append(r)
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort
    results = sorted(results, key=lambda x: x['metrics']['r2'], reverse=True)
    
    print("\n" + "="*60)
    print("DEEP LEARNING LEADERBOARD")
    print("="*60)
    print(f"{'#':<4}{'Model':<22}{'R2':>10}{'RMSE':>10}{'MAE':>10}{'Time':>8}")
    print("-"*60)
    
    for i, r in enumerate(results, 1):
        m = r['metrics']
        ok = "[OK]" if m['r2'] >= 0.95 else ""
        print(f"{i:<4}{r['model']:<22}{m['r2']:>10.4f}{m['rmse']:>10.4f}{m['mae']:>10.4f}{m['train_time']:>7.1f}s {ok}")
    
    # Compare with GradientBoosting
    print("\n" + "="*60)
    print("vs SKLEARN BASELINE")
    print("="*60)
    print(f"GradientBoosting: R2=0.9982, RMSE=0.0128, MAE=0.0080")
    
    if results:
        best = results[0]
        print(f"\nBest DL Model: {best['model']} R2={best['metrics']['r2']:.4f}")
    
    # Save
    output = project_root / "data/results"
    output.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output / f"dl_comparison_{ts}.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()