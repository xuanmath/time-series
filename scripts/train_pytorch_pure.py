"""
Pure PyTorch Model Comparison - LSTM vs GRU vs Transformer
No sklearn dependency for metrics
"""

import sys
import os
from pathlib import Path

# Set working directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import time
from datetime import datetime

print("=" * 60)
print("PyTorch Models Comparison: LSTM vs GRU vs Transformer")
print("=" * 60)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[Device] {device}")

# ============================================
# Load Data
# ============================================
print("\n[1/4] Loading data...")
df = pd.read_csv("data/raw/wind_power.csv")
print(f"    Shape: {df.shape}")

# Simple preprocessing with numpy
data = df['Power'].values.reshape(-1, 1)
data_min, data_max = data.min(), data.max()
data_scaled = (data - data_min) / (data_max - data_min)

# Split
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

print(f"    Train: {len(train_data)}, Test: {len(test_data)}")

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 24
X_train, y_train = create_sequences(train_data, seq_len)
X_test, y_test = create_sequences(test_data, seq_len)

print(f"    Sequences: {X_train.shape}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# Metrics functions (no sklearn)
def calc_metrics(y_true, y_pred):
    """Calculate metrics without sklearn."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # SMAPE
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'smape': smape}

def inverse_transform(data, data_min, data_max):
    """Inverse min-max scaling."""
    return data * (data_max - data_min) + data_min

results = {}

# ============================================
# LSTM Model Definition
# ============================================
class LSTMModel(nn.Module):
    def __init__(self, seq_len, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ============================================
# GRU Model Definition
# ============================================
class GRUModel(nn.Module):
    def __init__(self, seq_len, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ============================================
# Transformer Model Definition
# ============================================
class TransformerModel(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# ============================================
# Training Function
# ============================================
def train_model(model, X_train, y_train, epochs=50, batch_size=32, lr=1e-3, patience=10):
    """Train model with early stopping."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    return epoch + 1

# ============================================
# Train LSTM
# ============================================
print("\n[2/4] Training LSTM...")
lstm_model = LSTMModel(seq_len, hidden_size=64, num_layers=2)

start_time = time.time()
epochs_trained = train_model(lstm_model, X_train_t, y_train_t, epochs=50, patience=10)
lstm_time = time.time() - start_time

lstm_model.eval()
with torch.no_grad():
    lstm_pred = lstm_model(X_test_t).cpu().numpy()

lstm_pred_orig = inverse_transform(lstm_pred, data_min, data_max)
y_test_orig = inverse_transform(y_test, data_min, data_max)

lstm_metrics = calc_metrics(y_test_orig, lstm_pred_orig)
lstm_metrics['time'] = lstm_time
lstm_metrics['epochs'] = epochs_trained
results['LSTM'] = lstm_metrics

print(f"    LSTM - R2: {lstm_metrics['r2']:.4f}, RMSE: {lstm_metrics['rmse']:.4f}, Time: {lstm_time:.1f}s")

# ============================================
# Train GRU
# ============================================
print("\n[3/4] Training GRU...")
gru_model = GRUModel(seq_len, hidden_size=64, num_layers=2)

start_time = time.time()
epochs_trained = train_model(gru_model, X_train_t, y_train_t, epochs=50, patience=10)
gru_time = time.time() - start_time

gru_model.eval()
with torch.no_grad():
    gru_pred = gru_model(X_test_t).cpu().numpy()

gru_pred_orig = inverse_transform(gru_pred, data_min, data_max)

gru_metrics = calc_metrics(y_test_orig, gru_pred_orig)
gru_metrics['time'] = gru_time
gru_metrics['epochs'] = epochs_trained
results['GRU'] = gru_metrics

print(f"    GRU - R2: {gru_metrics['r2']:.4f}, RMSE: {gru_metrics['rmse']:.4f}, Time: {gru_time:.1f}s")

# ============================================
# Train Transformer
# ============================================
print("\n[4/4] Training Transformer...")
trans_model = TransformerModel(seq_len, d_model=64, nhead=4, num_layers=2)

start_time = time.time()
epochs_trained = train_model(trans_model, X_train_t, y_train_t, epochs=50, lr=1e-4, patience=15)
trans_time = time.time() - start_time

trans_model.eval()
with torch.no_grad():
    trans_pred = trans_model(X_test_t).cpu().numpy()

trans_pred_orig = inverse_transform(trans_pred, data_min, data_max)

trans_metrics = calc_metrics(y_test_orig, trans_pred_orig)
trans_metrics['time'] = trans_time
trans_metrics['epochs'] = epochs_trained
results['Transformer'] = trans_metrics

print(f"    Transformer - R2: {trans_metrics['r2']:.4f}, RMSE: {trans_metrics['rmse']:.4f}, Time: {trans_time:.1f}s")

# ============================================
# Print Comparison
# ============================================
print("\n" + "=" * 70)
print("PYTORCH MODELS COMPARISON RESULTS")
print("=" * 70)
print(f"{'Model':<15} {'R2':>10} {'RMSE':>12} {'MAE':>12} {'SMAPE':>10} {'Time(s)':>10}")
print("-" * 70)
for model, metrics in results.items():
    print(f"{model:<15} {metrics['r2']:>10.4f} {metrics['rmse']:>12.4f} {metrics['mae']:>12.4f} {metrics['smape']:>10.2f} {metrics['time']:>10.1f}")

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['r2'])
print("-" * 70)
print(f"[BEST] {best_model[0]} with R2 = {best_model[1]['r2']:.4f}")

# ============================================
# Save Results
# ============================================
output_dir = Path("logs")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"pytorch_comparison_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n[SAVE] Results saved to {output_file}")
print("\n[DONE] Comparison completed!")
