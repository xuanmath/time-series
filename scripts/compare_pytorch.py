"""
Simplified Model Comparison - 简化版模型对比

只使用 PyTorch 和 numpy，避免 sklearn 导入问题
"""

import sys
import json
import time
import numpy as np
import torch
import logging
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# Logging Setup
# ============================================================

def setup_logger(name, log_file=None):
    """Setup logger with file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(message)s')
    console.setFormatter(console_fmt)
    logger.addHandler(console)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model_info(model, name):
    """Get model architecture info."""
    trainable, total = count_parameters(model)
    layers = []
    for name_layer, module in model.named_modules():
        if len(name_layer) > 0 and not isinstance(module, torch.nn.Sequential):
            layers.append(f"  {name_layer}: {module.__class__.__name__}")
    return {
        'trainable_params': trainable,
        'total_params': total,
        'layers': '\n'.join(layers) if layers else '  (no submodules)'
    }


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")


# ============================================================
# Metrics (without sklearn)
# ============================================================

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / (ss_tot + 1e-10)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ============================================================
# Data Preparation (without sklearn)
# ============================================================

def create_sequences(X, y, seq_len, stride=1):
    """
    Create sequences with configurable stride.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        seq_len: Sequence length (window size)
        stride: Step size between windows (default=1, every timestep)
    
    Returns:
        X_seq: (n_sequences, seq_len, n_features)
        y_seq: (n_sequences,)
    """
    X_seq, y_seq = [], []
    for i in range(0, len(X) - seq_len, stride):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)


def create_multi_scale_sequences(X, y, seq_lens, stride=1):
    """
    Create sequences at multiple scales (different window lengths).
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        seq_lens: List of sequence lengths, e.g. [6, 12, 24, 48]
        stride: Step size between windows
    
    Returns:
        Dict with sequences for each scale, plus combined multi-scale features
    """
    sequences = {}
    for sl in seq_lens:
        X_seq, y_seq = create_sequences(X, y, sl, stride)
        sequences[f'scale_{sl}'] = {
            'X': X_seq,
            'y': y_seq,
            'seq_len': sl
        }
    
    # Create multi-scale features: concatenate features from different scales
    # Use the minimum seq_len as base, extract features from longer windows
    min_len = min(seq_lens)
    max_len = max(seq_lens)
    
    # For each position, gather statistics from different scales
    X_multi = []
    y_multi = []
    
    # We need to align sequences - use stride from the longest window perspective
    for i in range(0, len(X) - max_len, stride):
        # Get target at the end of longest window
        y_multi.append(y[i + max_len])
        
        # Build multi-scale features
        features = []
        for sl in seq_lens:
            window = X[i + max_len - sl:i + max_len]  # Last sl steps before prediction
            # Aggregate features: mean, std, min, max, last value
            features.extend([
                window.mean(axis=0),    # Mean of window
                window.std(axis=0),     # Std of window
                window.min(axis=0),     # Min
                window.max(axis=0),     # Max
                window[-1],             # Last value (most recent)
            ])
        X_multi.append(np.concatenate(features))
    
    sequences['multi_scale'] = {
        'X': np.array(X_multi),
        'y': np.array(y_multi),
        'seq_len': max_len,
        'n_features_per_scale': len(X[0]) * 5  # 5 aggregations per scale
    }
    
    return sequences


def prepare_data(data_path, seq_len=24, stride=1, multi_scale=False, 
                 seq_lens=None, log_preprocess=True):
    """
    Prepare data with configurable sliding window options.
    
    Args:
        data_path: Path to CSV data
        seq_len: Single window length (default 24 = 1 day for hourly data)
        stride: Step between windows (1 = every step, 6 = every 6 hours)
        multi_scale: Enable multi-scale windows
        seq_lens: List of window lengths for multi-scale (e.g. [6, 12, 24, 48])
        log_preprocess: Log preprocessing details
    
    Returns:
        Dict with prepared sequences and metadata
    """
    import pandas as pd
    
    df = pd.read_csv(data_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col not in ['Time', 'Power']]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['Power'].values.astype(np.float32)
    
    # Simple normalization
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-10
    y_mean, y_std = y.mean(), y.std() + 1e-10
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    # Split (time-series: use first 80% for train)
    split_idx = int(len(X_norm) * 0.8)
    X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
    y_train, y_test = y_norm[:split_idx], y_norm[split_idx:]
    
    if log_preprocess:
        print(f"[Preprocess] Raw data: {len(df)} samples")
        print(f"[Preprocess] Features: {len(feature_cols)} ({feature_cols})")
        print(f"[Preprocess] Train/Test: {len(X_train)}/{len(X_test)}")
        print(f"[Preprocess] Window: seq_len={seq_len}, stride={stride}")
    
    result = {
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'X_mean': X_mean.tolist(),
        'X_std': X_std.tolist(),
        'y_mean': float(y_mean),
        'y_std': float(y_std),
        'seq_len': seq_len,
        'stride': stride,
        'n_train_raw': len(X_train),
        'n_test_raw': len(X_test),
    }
    
    if multi_scale and seq_lens:
        # Multi-scale sequences
        if log_preprocess:
            print(f"[Preprocess] Multi-scale windows: {seq_lens}")
        
        train_seq = create_multi_scale_sequences(X_train, y_train, seq_lens, stride)
        test_seq = create_multi_scale_sequences(X_test, y_test, seq_lens, stride)
        
        # Use the primary scale (seq_len) for standard training
        primary_key = f'scale_{seq_len}'
        result['X_train_seq'] = train_seq[primary_key]['X']
        result['y_train_seq'] = train_seq[primary_key]['y']
        result['X_test_seq'] = test_seq[primary_key]['X']
        result['y_test_seq'] = test_seq[primary_key]['y']
        
        # Store all scales
        result['multi_scale'] = {
            'train': train_seq,
            'test': test_seq,
            'seq_lens': seq_lens
        }
        
        if log_preprocess:
            for sl in seq_lens:
                n_train = len(train_seq[f'scale_{sl}']['X'])
                n_test = len(test_seq[f'scale_{sl}']['X'])
                print(f"[Preprocess] Scale {sl}: train={n_train}, test={n_test}")
    else:
        # Single-scale sequences
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len, stride)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len, stride)
        
        result['X_train_seq'] = X_train_seq
        result['y_train_seq'] = y_train_seq
        result['X_test_seq'] = X_test_seq
        result['y_test_seq'] = y_test_seq
        
        if log_preprocess:
            print(f"[Preprocess] Sequences: train={len(X_train_seq)}, test={len(X_test_seq)}")
            print(f"[Preprocess] Sequence shape: {X_train_seq.shape}")
    
    return result


# ============================================================
# Models
# ============================================================

class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SimpleGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class SimpleTransformer(torch.nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj = torch.nn.Linear(input_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = torch.nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.proj(x)
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


class SimpleCNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(64, hidden_size, 2, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (B, T, F)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class MLP(torch.nn.Module):
    def __init__(self, input_size, seq_len=24, hidden_size=128):
        super().__init__()
        self.flatten_size = input_size * seq_len
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.flatten_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.net(x)


# ============================================================
# Training
# ============================================================

def train_model(data, model_class, name, epochs=20, batch_size=64, lr=0.001, logger=None):
    if logger:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
    
    X_train = torch.FloatTensor(data['X_train_seq']).to(DEVICE)
    y_train = torch.FloatTensor(data['y_train_seq']).to(DEVICE)
    X_test = torch.FloatTensor(data['X_test_seq']).to(DEVICE)
    y_test = data['y_test_seq'].flatten()
    
    model = model_class(input_size=data['n_features']).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Log model info
    model_info = get_model_info(model, name)
    if logger:
        logger.info(f"Model Architecture: {name}")
        logger.info(f"Trainable Parameters: {model_info['trainable_params']:,}")
        logger.info(f"Total Parameters: {model_info['total_params']:,}")
        logger.info(f"Learning Rate: {lr}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Training Samples: {len(X_train)}")
        logger.debug(f"Layers:\n{model_info['layers']}")
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    t0 = time.time()
    model.train()
    
    # Track epoch history
    epoch_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log every epoch
        if logger:
            logger.debug(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
        
        # Log every 5 epochs to console
        if (epoch + 1) % 5 == 0:
            if logger:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")
            else:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")
        
        # Record epoch history (convert to native types)
        epoch_history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'lr': float(current_lr),
            'trainable_params': int(model_info['trainable_params'])
        })
    
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
    
    if logger:
        logger.info(f"Final Results: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        logger.info(f"Training Time: {train_time:.1f}s")
    else:
        print(f"R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        print(f"Time: {train_time:.1f}s")
    
    return {
        'model': name, 
        'metrics': metrics,
        'epoch_history': epoch_history,
        'model_info': {
            'trainable_params': model_info['trainable_params'],
            'total_params': model_info['total_params'],
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': epochs
        }
    }


# ============================================================
# Main
# ============================================================

def main():
    # Setup logger
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{ts}.log"
    
    logger = setup_logger('model_training', str(log_file))
    
    logger.info("\n" + "#"*50)
    logger.info("# PYTORCH MODEL COMPARISON")
    logger.info("#"*50)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Log File: {log_file}")
    
    data_path = project_root / "dataset/Location1.csv"
    logger.info(f"\nLoading: {data_path}")
    
    data = prepare_data(str(data_path), seq_len=24)
    logger.info(f"Train Shape: {data['X_train_seq'].shape}")
    logger.info(f"Test Shape: {data['X_test_seq'].shape}")
    logger.info(f"Features: {data['n_features']}")
    
    results = []
    
    models = [
        (MLP, "MLP"),
        (SimpleLSTM, "LSTM"),
        (SimpleGRU, "GRU"),
        (SimpleTransformer, "Transformer"),
        (SimpleCNN, "CNN-GRU"),
    ]
    
    for model_class, name in models:
        try:
            r = train_model(data, model_class, name, epochs=20, logger=logger)
            results.append(r)
        except Exception as e:
            logger.error(f"[FAIL] {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort
    results = sorted(results, key=lambda x: x['metrics']['r2'], reverse=True)
    
    logger.info("\n" + "="*50)
    logger.info("LEADERBOARD")
    logger.info("="*50)
    leaderboard_str = f"{'#':<4}{'Model':<15}{'R2':>10}{'RMSE':>10}{'MAE':>10}{'Time':>8}"
    logger.info(leaderboard_str)
    logger.info("-"*50)
    
    for i, r in enumerate(results, 1):
        m = r['metrics']
        ok = "[OK]" if m['r2'] >= 0.95 else ""
        row = f"{i:<4}{r['model']:<15}{m['r2']:>10.4f}{m['rmse']:>10.4f}{m['mae']:>10.4f}{m['train_time']:>7.1f}s {ok}"
        logger.info(row)
    
    # Save results JSON
    output = project_root / "data/results"
    output.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        return obj
    
    results_native = convert_to_native(results)
    
    result_file = output / f"pytorch_comparison_{ts}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results_native, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved: {result_file}")
    
    # Save epoch history to separate CSV for analysis
    history_file = output / f"epoch_history_{ts}.csv"
    with open(history_file, 'w', encoding='utf-8') as f:
        f.write("model,epoch,loss,lr,trainable_params\n")
        for r in results:
            for h in r.get('epoch_history', []):
                f.write(f"{r['model']},{h['epoch']},{h['loss']:.6f},{h['lr']:.6f},{h['trainable_params']}\n")
    logger.info(f"Epoch history saved: {history_file}")
    
    logger.info(f"\nLog file: {log_file}")
    
    if results:
        best = results[0]
        logger.info(f"\n[BEST] Model: {best['model']} | R2={best['metrics']['r2']:.4f}")
    
    print(f"\n[DONE] Training complete! Log saved: {log_file}")
    print(f"[DONE] Results saved: {result_file}")
    print(f"[DONE] Epoch history: {history_file}")


if __name__ == "__main__":
    main()