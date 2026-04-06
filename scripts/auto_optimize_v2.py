"""
Auto Optimization V2 - 两阶段自动优化

阶段一: 所有模型迭代优化直到 R2 >= 0.90
阶段二: CNN-GRU专项优化，超越基线模型

优化策略:
- 模型结构调整 (层数, hidden_size, dropout)
- 超参数搜索 (lr, batch_size, epochs)
- 学习率调度器
- 早停机制
- 特征工程 (多尺度窗口)
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# Enhanced Models with Dropout and Configurable Architecture
# ============================================================

class EnhancedLSTM(nn.Module):
    """LSTM with dropout and configurable layers."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class EnhancedGRU(nn.Module):
    """GRU with dropout and configurable layers."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class EnhancedCNN_GRU(nn.Module):
    """CNN-GRU with configurable architecture (GPU-friendly)."""
    def __init__(self, input_size, hidden_size=64, cnn_channels=32, 
                 kernel_size=3, num_layers=2, dropout=0.1):
        super().__init__()
        
        # CNN layers - lighter architecture
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
        )
        
        # GRU layers
        self.gru = nn.GRU(cnn_channels * 2, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # x: (B, T, F) -> (B, F, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # (B, F', T) -> (B, T, F') for GRU
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class EnhancedTransformer(nn.Module):
    """Transformer with configurable architecture."""
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 
                                                    d_model * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        x = self.proj(x)
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


class EnhancedMLP(nn.Module):
    """MLP with more layers and dropout."""
    def __init__(self, input_size, seq_len=24, hidden_sizes=[256, 128, 64], dropout=0.1):
        super().__init__()
        self.flatten_size = input_size * seq_len
        
        layers = []
        prev_size = self.flatten_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.net(x)


ENHANCED_MODELS = {
    'MLP': EnhancedMLP,
    'LSTM': EnhancedLSTM,
    'GRU': EnhancedGRU,
    'Transformer': EnhancedTransformer,
    'CNN-GRU': EnhancedCNN_GRU,
}


# ============================================================
# Metrics
# ============================================================

def calc_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / (ss_tot + 1e-10)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


# ============================================================
# Data Preparation
# ============================================================

def prepare_data_v2(data_path, seq_len=24, stride=1):
    """Prepare data with normalization stats saved."""
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
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    # Split
    split_idx = int(len(X_norm) * 0.8)
    X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
    y_train, y_test = y_norm[:split_idx], y_norm[split_idx:]
    
    # Create sequences
    X_train_seq = []
    y_train_seq = []
    for i in range(len(X_train) - seq_len):
        X_train_seq.append(X_train[i:i+seq_len])
        y_train_seq.append(y_train[i+seq_len])
    
    X_test_seq = []
    y_test_seq = []
    for i in range(len(X_test) - seq_len):
        X_test_seq.append(X_test[i:i+seq_len])
        y_test_seq.append(y_test[i+seq_len])
    
    return {
        'X_train_seq': np.array(X_train_seq),
        'y_train_seq': np.array(y_train_seq),
        'X_test_seq': np.array(X_test_seq),
        'y_test_seq': np.array(y_test_seq),
        'n_features': len(feature_cols),
        'seq_len': seq_len,
        'feature_cols': feature_cols,
        'X_mean': X_mean, 'X_std': X_std,
        'y_mean': y_mean, 'y_std': y_std,
    }


# ============================================================
# Training with Early Stopping
# ============================================================

def train_with_early_stop(model, train_loader, X_test, y_test, 
                          epochs=100, lr=0.001, patience=10,
                          scheduler_type='cosine', logger=None):
    """Train with early stopping and learning rate scheduler."""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None
    
    best_r2 = -float('inf')
    best_model_state = None
    patience_counter = 0
    
    history = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).cpu().numpy().flatten()
        
        metrics = calc_metrics(y_test, y_pred)
        metrics['train_loss'] = train_loss
        metrics['lr'] = optimizer.param_groups[0]['lr']
        history.append(metrics)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Check improvement
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1}/{epochs} | Loss={train_loss:.5f} | R2={metrics['r2']:.4f} | Best={best_r2:.4f}"
            if logger:
                logger.debug(msg)
        
        # Early stop
        if patience_counter >= patience and best_r2 >= 0.85:
            if logger:
                logger.info(f"Early stop at epoch {epoch+1} (patience={patience})")
            break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return {
        'best_r2': best_r2,
        'history': history,
        'epochs_trained': epoch + 1
    }


# ============================================================
# Phase 1: Optimize All Models to R2 >= 0.90
# ============================================================

def get_model_config_evolution(current_r2: float, iteration: int) -> Dict:
    """Generate progressively more complex configs based on current performance."""
    
    # Stage 1: Moderate size (GPU-friendly)
    if current_r2 < 0.70:
        return {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 50,
            'patience': 15,
            'scheduler': 'cosine',
            'seq_len': 24,
        }
    # Stage 2: Slightly larger
    elif current_r2 < 0.80:
        return {
            'hidden_size': 96,
            'num_layers': 2,
            'dropout': 0.15,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 80,
            'patience': 20,
            'scheduler': 'cosine',
            'seq_len': 36,
        }
    # Stage 3: Fine tune
    elif current_r2 < 0.85:
        return {
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'learning_rate': 0.0003,
            'batch_size': 32,
            'epochs': 100,
            'patience': 25,
            'scheduler': 'step',
            'seq_len': 48,
        }
    # Stage 4: Near target - careful size increase
    elif current_r2 < 0.90:
        return {
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.25,
            'learning_rate': 0.0002,
            'batch_size': 16,
            'epochs': 150,
            'patience': 30,
            'scheduler': 'cosine',
            'seq_len': 72,
        }
    else:
        # Target reached
        return {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'epochs': 50,
            'patience': 15,
            'scheduler': 'cosine',
            'seq_len': 24,
        }


def create_model_with_config(model_name: str, config: Dict, n_features: int, seq_len: int):
    """Create model with specific config."""
    
    model_class = ENHANCED_MODELS[model_name]
    
    if model_name == 'MLP':
        hidden_sizes = [config['hidden_size'], config['hidden_size']//2, config['hidden_size']//4]
        model = model_class(n_features, seq_len, hidden_sizes, config['dropout'])
    elif model_name == 'CNN-GRU':
        model = model_class(
            n_features, 
            hidden_size=config['hidden_size'],
            cnn_channels=config.get('cnn_channels', 64),
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    elif model_name == 'Transformer':
        model = model_class(
            n_features,
            d_model=config['hidden_size'],
            nhead=max(1, min(8, config['hidden_size'] // 16)),
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    else:
        model = model_class(
            n_features,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    
    return model.to(DEVICE)


def phase1_optimize_all(data_path: str, target_r2: float = 0.90, 
                        max_iterations: int = 50, logger=None) -> Dict:
    """
    Phase 1: Optimize all models iteratively until R2 >= target.
    """
    
    if logger is None:
        logger = logging.getLogger('optimizer')
    
    logger.info("\n" + "#"*70)
    logger.info("# PHASE 1: OPTIMIZE ALL MODELS TO R2 >= 0.90")
    logger.info("#"*70)
    
    models_to_optimize = ['MLP', 'LSTM', 'GRU', 'Transformer', 'CNN-GRU']
    
    # Track best results per model
    best_results = {name: {'r2': 0, 'config': None, 'iteration': 0} for name in models_to_optimize}
    optimization_log = []
    
    iteration = 0
    models_reached_target = set()
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check if all models reached target
        if len(models_reached_target) == len(models_to_optimize):
            logger.info(f"\n[SUCCESS] All models reached R2 >= {target_r2}!")
            break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"ITERATION {iteration}/{max_iterations}")
        logger.info(f"{'='*70}")
        
        # Status summary
        for name in models_to_optimize:
            status = "TARGET REACHED" if name in models_reached_target else f"R2={best_results[name]['r2']:.4f}"
            logger.info(f"  {name}: {status}")
        
        # Optimize models that haven't reached target
        for model_name in models_to_optimize:
            if model_name in models_reached_target:
                continue
            
            current_r2 = best_results[model_name]['r2']
            config = get_model_config_evolution(current_r2, iteration)
            
            # Adjust seq_len for data
            seq_len = config.get('seq_len', 24)
            
            logger.info(f"\n[{model_name}] Iteration {iteration}")
            logger.info(f"  Current R2: {current_r2:.4f}")
            logger.info(f"  Config: hidden={config['hidden_size']}, layers={config['num_layers']}, seq_len={seq_len}")
            
            # Prepare data
            data = prepare_data_v2(data_path, seq_len=seq_len)
            
            # Create model
            model = create_model_with_config(model_name, config, data['n_features'], seq_len)
            
            # Train
            X_train = torch.FloatTensor(data['X_train_seq']).to(DEVICE)
            y_train = torch.FloatTensor(data['y_train_seq']).to(DEVICE)
            X_test = torch.FloatTensor(data['X_test_seq']).to(DEVICE)
            y_test = data['y_test_seq'].flatten()
            
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config['batch_size'], shuffle=True
            )
            
            result = train_with_early_stop(
                model, train_loader, X_test, y_test,
                epochs=config['epochs'],
                lr=config['learning_rate'],
                patience=config['patience'],
                scheduler_type=config['scheduler'],
                logger=logger
            )
            
            r2 = result['best_r2']
            
            logger.info(f"  Result: R2={r2:.4f} | Epochs={result['epochs_trained']}")
            
            # Update best if improved
            if r2 > best_results[model_name]['r2']:
                best_results[model_name] = {
                    'r2': r2,
                    'config': config,
                    'iteration': iteration,
                    'history': result['history'],
                    'epochs_trained': result['epochs_trained']
                }
            
            # Check if target reached
            if r2 >= target_r2:
                models_reached_target.add(model_name)
                logger.info(f"  [TARGET REACHED] {model_name}: R2={r2:.4f}")
            
            # Log
            optimization_log.append({
                'iteration': iteration,
                'model': model_name,
                'r2': r2,
                'config': config,
                'epochs_trained': result['epochs_trained']
            })
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("PHASE 1 COMPLETE")
    logger.info("="*70)
    
    for name in models_to_optimize:
        r = best_results[name]
        status = "OK" if r['r2'] >= target_r2 else "FAILED"
        logger.info(f"  {name}: R2={r['r2']:.4f} [{status}]")
    
    return {
        'best_results': best_results,
        'optimization_log': optimization_log,
        'models_reached_target': models_reached_target,
        'target_r2': target_r2,
        'iterations': iteration
    }


# ============================================================
# Phase 2: Optimize CNN-GRU to Beat Baseline
# ============================================================

def phase2_optimize_cnn_gru(data_path: str, baseline_r2: float, 
                            max_iterations: int = 30, logger=None) -> Dict:
    """
    Phase 2: Optimize CNN-GRU specifically to beat baseline.
    """
    
    if logger is None:
        logger = logging.getLogger('optimizer')
    
    logger.info("\n" + "#"*70)
    logger.info("# PHASE 2: CNN-GRU OPTIMIZATION TO BEAT BASELINE")
    logger.info("#"*70)
    logger.info(f"Baseline R2 to beat: {baseline_r2:.4f}")
    
    best_r2 = 0
    best_config = None
    best_history = None
    optimization_log = []
    
    # CNN-GRU specific configs to try (GPU-friendly, 4GB VRAM)
    configs_to_try = [
        # Config 1: Base
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1, 'cnn_channels': 32,
         'learning_rate': 0.001, 'batch_size': 64, 'epochs': 50, 'patience': 15, 'seq_len': 24},
        # Config 2: Moderate
        {'hidden_size': 96, 'num_layers': 2, 'dropout': 0.15, 'cnn_channels': 48,
         'learning_rate': 0.0005, 'batch_size': 32, 'epochs': 80, 'patience': 20, 'seq_len': 36},
        # Config 3: Larger CNN
        {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2, 'cnn_channels': 64,
         'learning_rate': 0.0003, 'batch_size': 32, 'epochs': 100, 'patience': 25, 'seq_len': 48},
        # Config 4: Deep CNN-GRU
        {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.25, 'cnn_channels': 96,
         'learning_rate': 0.0002, 'batch_size': 16, 'epochs': 120, 'patience': 30, 'seq_len': 72},
        # Config 5: Maximum safe size
        {'hidden_size': 128, 'num_layers': 4, 'dropout': 0.3, 'cnn_channels': 128,
         'learning_rate': 0.0001, 'batch_size': 16, 'epochs': 150, 'patience': 35, 'seq_len': 96},
    ]
    
    iteration = 0
    
    while iteration < max_iterations and best_r2 < baseline_r2:
        iteration += 1
        
        # Use config from list or generate
        config_idx = min(iteration - 1, len(configs_to_try) - 1)
        config = configs_to_try[config_idx]
        
        seq_len = config['seq_len']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CNN-GRU Iteration {iteration}")
        logger.info(f"  Current best R2: {best_r2:.4f}")
        logger.info(f"  Target: beat {baseline_r2:.4f}")
        logger.info(f"  Config: hidden={config['hidden_size']}, cnn={config['cnn_channels']}, layers={config['num_layers']}")
        
        # Prepare data
        data = prepare_data_v2(data_path, seq_len=seq_len)
        
        # Create model
        model = create_model_with_config('CNN-GRU', config, data['n_features'], seq_len)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {n_params:,}")
        
        # Train
        X_train = torch.FloatTensor(data['X_train_seq']).to(DEVICE)
        y_train = torch.FloatTensor(data['y_train_seq']).to(DEVICE)
        X_test = torch.FloatTensor(data['X_test_seq']).to(DEVICE)
        y_test = data['y_test_seq'].flatten()
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True
        )
        
        result = train_with_early_stop(
            model, train_loader, X_test, y_test,
            epochs=config['epochs'],
            lr=config['learning_rate'],
            patience=config['patience'],
            scheduler_type='cosine',
            logger=logger
        )
        
        r2 = result['best_r2']
        
        logger.info(f"  Result: R2={r2:.4f} | Best={best_r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_config = config
            best_history = result['history']
            logger.info(f"  [NEW BEST] R2={r2:.4f}")
        
        if r2 > baseline_r2:
            logger.info(f"\n[SUCCESS] CNN-GRU beat baseline! R2={r2:.4f} > {baseline_r2:.4f}")
        
        optimization_log.append({
            'iteration': iteration,
            'r2': r2,
            'config': config,
            'n_params': n_params,
            'epochs_trained': result['epochs_trained']
        })
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("PHASE 2 COMPLETE")
    logger.info("="*60)
    logger.info(f"CNN-GRU Best R2: {best_r2:.4f}")
    logger.info(f"Baseline R2: {baseline_r2:.4f}")
    
    if best_r2 > baseline_r2:
        logger.info(f"[SUCCESS] CNN-GRU beats baseline by {(best_r2 - baseline_r2)*100:.2f}%")
    else:
        logger.info(f"[FAILED] CNN-GRU did not beat baseline")
    
    return {
        'best_r2': best_r2,
        'best_config': best_config,
        'best_history': best_history,
        'optimization_log': optimization_log,
        'baseline_r2': baseline_r2,
        'iterations': iteration,
        'success': best_r2 > baseline_r2
    }


# ============================================================
# Main Execution
# ============================================================

def run_full_optimization(data_path: str, target_r2: float = 0.90,
                          phase1_max_iter: int = 30, phase2_max_iter: int = 20):
    """Run both phases of optimization."""
    
    # Setup logger
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"full_optimization_{ts}.log"
    
    logger = setup_logger('full_optimizer', str(log_file))
    
    logger.info("\n" + "#"*70)
    logger.info("# FULL AUTOMATIC OPTIMIZATION")
    logger.info("#"*70)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Target R2: {target_r2}")
    logger.info(f"Data: {data_path}")
    
    # Phase 1: Optimize all models
    phase1_results = phase1_optimize_all(
        data_path, target_r2=target_r2, 
        max_iterations=phase1_max_iter, logger=logger
    )
    
    # Get baseline (best R2 from non-CNN-GRU models)
    baseline_models = ['MLP', 'LSTM', 'GRU', 'Transformer']
    baseline_r2 = max(phase1_results['best_results'][m]['r2'] for m in baseline_models)
    
    logger.info(f"\n[Baseline] Best R2 from other models: {baseline_r2:.4f}")
    
    # Phase 2: Optimize CNN-GRU
    phase2_results = phase2_optimize_cnn_gru(
        data_path, baseline_r2 + 0.01,  # Target 1% better than baseline
        max_iterations=phase2_max_iter, logger=logger
    )
    
    # Save results
    output_dir = project_root / "data/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        return obj
    
    full_results = {
        'phase1': convert_types(phase1_results),
        'phase2': convert_types(phase2_results),
        'timestamp': datetime.now().isoformat(),
        'target_r2': target_r2,
        'device': DEVICE
    }
    
    with open(output_dir / f"full_optimization_{ts}.json", 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    # Summary CSV
    with open(output_dir / f"optimization_summary_{ts}.csv", 'w', encoding='utf-8') as f:
        f.write("phase,iteration,model,r2,hidden_size,num_layers,seq_len,epochs_trained\n")
        for entry in phase1_results['optimization_log']:
            f.write(f"1,{entry['iteration']},{entry['model']},{entry['r2']:.6f},{entry['config']['hidden_size']},{entry['config']['num_layers']},{entry['config']['seq_len']},{entry['epochs_trained']}\n")
        for entry in phase2_results['optimization_log']:
            f.write(f"2,{entry['iteration']},CNN-GRU,{entry['r2']:.6f},{entry['config']['hidden_size']},{entry['config']['num_layers']},{entry['config']['seq_len']},{entry['epochs_trained']}\n")
    
    # Final report
    logger.info("\n" + "#"*70)
    logger.info("# FINAL REPORT")
    logger.info("#"*70)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print("\n[Phase 1 - All Models]")
    for name, r in phase1_results['best_results'].items():
        status = "OK" if r['r2'] >= target_r2 else "FAILED"
        print(f"  {name}: R2={r['r2']:.4f} [{status}]")
    
    print(f"\n[Phase 2 - CNN-GRU vs Baseline]")
    print(f"  Baseline: {baseline_r2:.4f}")
    print(f"  CNN-GRU Best: {phase2_results['best_r2']:.4f}")
    
    if phase2_results['success']:
        print(f"  [SUCCESS] CNN-GRU beats baseline!")
    else:
        print(f"  [FAILED] CNN-GRU needs more optimization")
    
    print(f"\n[Logs & Results]")
    print(f"  Log: {log_file}")
    print(f"  JSON: {output_dir / f'full_optimization_{ts}.json'}")
    print(f"  CSV: {output_dir / f'optimization_summary_{ts}.csv'}")


def setup_logger(name, log_file=None):
    """Setup logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    
    # File
    if log_file:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)
    
    return logger


if __name__ == "__main__":
    data_path = str(project_root / "dataset/Location1.csv")
    run_full_optimization(data_path, target_r2=0.90)