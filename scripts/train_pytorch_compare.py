"""
Simple PyTorch Model Training - LSTM, GRU, Transformer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime

print("=" * 60)
print("PyTorch Models Comparison: LSTM vs GRU vs Transformer")
print("=" * 60)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[Device] {device}")

# Load data
print("\n[1/3] Loading data...")
df = pd.read_csv("data/raw/wind_power.csv")
print(f"    Shape: {df.shape}")

# Simple preprocessing
from sklearn.preprocessing import MinMaxScaler

data = df['Power'].values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

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

# Results storage
results = {}

# ============================================
# LSTM Model
# ============================================
print("\n[2/3] Training LSTM...")

from src.models.lstm import LSTMModel

lstm_model = LSTMModel(
    seq_len=seq_len,
    pred_len=1,
    hidden_size=64,
    num_layers=2,
    dropout=0.1,
    learning_rate=1e-3,
    epochs=50,
    batch_size=32
)

import time
start = time.time()
y_train_series = pd.Series(train_data.flatten())
lstm_model.fit(y_train_series, verbose=0, early_stopping_patience=10)
lstm_time = time.time() - start

# Predict
last_values = train_data[-seq_len:]
lstm_pred = lstm_model.predict(steps=len(test_data), last_values=last_values)

# Inverse transform
lstm_pred_orig = scaler.inverse_transform(lstm_pred.reshape(-1, 1))
y_test_orig = scaler.inverse_transform(y_test)

# Align lengths
min_len = min(len(lstm_pred_orig), len(y_test_orig))
lstm_pred_orig = lstm_pred_orig[:min_len]
y_test_aligned = y_test_orig[:min_len]

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
lstm_mae = mean_absolute_error(y_test_aligned, lstm_pred_orig)
lstm_rmse = np.sqrt(mean_squared_error(y_test_aligned, lstm_pred_orig))
lstm_r2 = r2_score(y_test_aligned, lstm_pred_orig)

results['LSTM'] = {
    'mae': lstm_mae,
    'rmse': lstm_rmse,
    'r2': lstm_r2,
    'time': lstm_time
}
print(f"    LSTM - R2: {lstm_r2:.4f}, RMSE: {lstm_rmse:.4f}, Time: {lstm_time:.1f}s")

# ============================================
# GRU Model
# ============================================
print("\n[2/3] Training GRU...")

from src.models.gru import GRUForecaster

gru_model = GRUForecaster(
    seq_len=seq_len,
    pred_len=1,
    hidden_size=64,
    num_layers=2,
    dropout=0.1,
    learning_rate=1e-3,
    epochs=50,
    batch_size=32
)

start = time.time()
gru_model.fit(train_data.flatten(), verbose=0, early_stopping_patience=10)
gru_time = time.time() - start

# Predict
gru_pred = gru_model.predict(steps=len(test_data), last_values=train_data[-seq_len:])

# Inverse transform
gru_pred_orig = scaler.inverse_transform(gru_pred.reshape(-1, 1))

# Metrics - align lengths
min_len = min(len(gru_pred_orig), len(y_test_orig))
gru_pred_orig = gru_pred_orig[:min_len]
y_test_aligned = y_test_orig[:min_len]

gru_mae = mean_absolute_error(y_test_aligned, gru_pred_orig)
gru_rmse = np.sqrt(mean_squared_error(y_test_aligned, gru_pred_orig))
gru_r2 = r2_score(y_test_aligned, gru_pred_orig)

results['GRU'] = {
    'mae': gru_mae,
    'rmse': gru_rmse,
    'r2': gru_r2,
    'time': gru_time
}
print(f"    GRU - R2: {gru_r2:.4f}, RMSE: {gru_rmse:.4f}, Time: {gru_time:.1f}s")

# ============================================
# Transformer Model
# ============================================
print("\n[3/3] Training Transformer...")

from src.models.transformer import TransformerForecaster

trans_model = TransformerForecaster(
    seq_len=seq_len,
    pred_len=1,
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    learning_rate=1e-4,
    epochs=50,
    batch_size=32
)

start = time.time()
trans_model.fit(X_train, y_train, verbose=0, early_stopping_patience=10)
trans_time = time.time() - start

# Predict
trans_pred = trans_model.predict(X_test).flatten()

# Inverse transform
trans_pred_orig = scaler.inverse_transform(trans_pred.reshape(-1, 1))

# Metrics - align lengths
min_len = min(len(trans_pred_orig), len(y_test_orig))
trans_pred_orig = trans_pred_orig[:min_len]
y_test_aligned = y_test_orig[:min_len]

trans_mae = mean_absolute_error(y_test_aligned, trans_pred_orig)
trans_rmse = np.sqrt(mean_squared_error(y_test_aligned, trans_pred_orig))
trans_r2 = r2_score(y_test_aligned, trans_pred_orig)

results['Transformer'] = {
    'mae': trans_mae,
    'rmse': trans_rmse,
    'r2': trans_r2,
    'time': trans_time
}
print(f"    Transformer - R2: {trans_r2:.4f}, RMSE: {trans_rmse:.4f}, Time: {trans_time:.1f}s")

# ============================================
# Print Comparison
# ============================================
print("\n" + "=" * 70)
print("COMPARISON RESULTS (PyTorch Models)")
print("=" * 70)
print(f"{'Model':<15} {'R2':>10} {'RMSE':>12} {'MAE':>12} {'Time(s)':>10}")
print("-" * 70)
for model, metrics in results.items():
    print(f"{model:<15} {metrics['r2']:>10.4f} {metrics['rmse']:>12.4f} {metrics['mae']:>12.4f} {metrics['time']:>10.1f}")

# Save results
output_dir = Path("logs")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(output_dir / f"pytorch_comparison_{timestamp}.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[SAVE] Results saved to logs/pytorch_comparison_{timestamp}.json")
print("\n[DONE] PyTorch models comparison completed!")
