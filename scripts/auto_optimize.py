"""
Auto Optimization - 自动优化训练

支持:
- 超参数网格搜索 (learning_rate, batch_size, hidden_size, seq_len)
- 多尺度窗口对比
- 自动选择最佳配置
- 优化历史记录
"""

import sys
import json
import time
import itertools
import numpy as np
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.compare_pytorch import (
    prepare_data, 
    train_model, 
    setup_logger,
    SimpleLSTM, SimpleGRU, SimpleTransformer, SimpleCNN, MLP,
    DEVICE
)


# ============================================================
# Optimization Configurations
# ============================================================

DEFAULT_SEARCH_SPACE = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128],
    'hidden_size': [32, 64, 128],
    'seq_len': [12, 24, 48],
    'epochs': [20, 30],
}

MODEL_CLASSES = {
    'MLP': MLP,
    'LSTM': SimpleLSTM,
    'GRU': SimpleGRU,
    'Transformer': SimpleTransformer,
    'CNN-GRU': SimpleCNN,
}


# ============================================================
# Model Variants with Configurable Hidden Size
# ============================================================

class ConfigurableLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class ConfigurableGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class ConfigurableMLP(torch.nn.Module):
    def __init__(self, input_size, seq_len=24, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
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


CONFIGURABLE_MODELS = {
    'MLP': ConfigurableMLP,
    'LSTM': ConfigurableLSTM,
    'GRU': ConfigurableGRU,
}


# ============================================================
# Optimization Functions
# ============================================================

def generate_configs(search_space: Dict) -> List[Dict]:
    """Generate all combinations from search space."""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    combinations = list(itertools.product(*values))
    
    configs = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        configs.append(config)
    
    return configs


def train_with_config(data: Dict, model_name: str, config: Dict, 
                      logger: logging.Logger) -> Dict:
    """Train model with specific configuration."""
    
    # Prepare data with configured seq_len
    seq_len = config.get('seq_len', 24)
    
    # If data already has the right seq_len, use it
    # Otherwise need to re-prepare
    if data['seq_len'] != seq_len:
        # Use stored normalization params
        X_mean = np.array(data['X_mean'])
        X_std = np.array(data['X_std'])
        y_mean = data['y_mean']
        y_std = data['y_std']
        
        # Re-create sequences with new seq_len
        # This requires original data - simplified here
        logger.warning(f"seq_len mismatch: data has {data['seq_len']}, config wants {seq_len}")
    
    X_train = torch.FloatTensor(data['X_train_seq']).to(DEVICE)
    y_train = torch.FloatTensor(data['y_train_seq']).to(DEVICE)
    X_test = torch.FloatTensor(data['X_test_seq']).to(DEVICE)
    y_test = data['y_test_seq'].flatten()
    
    # Create model with config
    hidden_size = config.get('hidden_size', 64)
    model_class = CONFIGURABLE_MODELS.get(model_name, CONFIGURABLE_MODELS['LSTM'])
    
    # Handle MLP which needs seq_len
    if model_name == 'MLP':
        model = model_class(
            input_size=data['n_features'],
            seq_len=data['seq_len'],
            hidden_size=hidden_size
        ).to(DEVICE)
    else:
        model = model_class(
            input_size=data['n_features'],
            hidden_size=hidden_size
        ).to(DEVICE)
    
    # Training params
    lr = config.get('learning_rate', 0.001)
    batch_size = config.get('batch_size', 64)
    epochs = config.get('epochs', 20)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    t0 = time.time()
    model.train()
    
    epoch_history = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        epoch_history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'lr': float(lr)
        })
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            logger.debug(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")
    
    train_time = time.time() - t0
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()
    
    # Calculate metrics
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - (ss_res / (ss_tot + 1e-10)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    result = {
        'model': model_name,
        'config': config,
        'metrics': {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'train_time': train_time,
            'best_loss': float(best_loss)
        },
        'epoch_history': epoch_history,
        'model_info': {
            'trainable_params': trainable,
            'hidden_size': hidden_size,
        }
    }
    
    logger.info(f"  R2={r2:.4f}, RMSE={rmse:.4f}, Time={train_time:.1f}s")
    
    return result


def run_optimization(data_path: str, 
                     models: List[str] = ['MLP', 'LSTM', 'GRU'],
                     search_space: Dict = None,
                     max_configs: int = 50,
                     logger: logging.Logger = None) -> Dict:
    """
    Run optimization across models and configurations.
    
    Args:
        data_path: Path to data CSV
        models: List of model names to optimize
        search_space: Dict of parameter ranges
        max_configs: Maximum configs to try per model
        logger: Logger instance
    
    Returns:
        Dict with optimization results and best configs
    """
    
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
    
    if logger is None:
        logger = setup_logger('optimizer')
    
    logger.info("\n" + "#"*60)
    logger.info("# AUTO OPTIMIZATION")
    logger.info("#"*60)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Models: {models}")
    logger.info(f"Search space: {search_space}")
    
    # Generate configs
    configs = generate_configs(search_space)
    logger.info(f"Total configs generated: {len(configs)}")
    
    # Limit configs if too many
    if len(configs) > max_configs:
        # Random sample
        np.random.seed(42)
        indices = np.random.choice(len(configs), max_configs, replace=False)
        configs = [configs[i] for i in indices]
        logger.info(f"Limited to {max_configs} configs (random sample)")
    
    # Prepare base data (will be re-prepared for different seq_lens)
    base_data = prepare_data(data_path, seq_len=24, stride=1, log_preprocess=False)
    
    # Pre-prepare data for each seq_len
    seq_lens = search_space.get('seq_len', [24])
    data_variants = {}
    for sl in seq_lens:
        data_variants[sl] = prepare_data(data_path, seq_len=sl, stride=1, log_preprocess=False)
        logger.info(f"Prepared data for seq_len={sl}: shape={data_variants[sl]['X_train_seq'].shape}")
    
    # Run optimization
    all_results = []
    optimization_history = []
    
    total_runs = len(models) * len(configs)
    logger.info(f"\nTotal optimization runs: {total_runs}")
    logger.info("="*60)
    
    run_idx = 0
    for model_name in models:
        logger.info(f"\n[Model: {model_name}]")
        
        for config in configs:
            run_idx += 1
            seq_len = config.get('seq_len', 24)
            
            # Get appropriate data
            data = data_variants.get(seq_len, base_data)
            
            # Skip if MLP and seq_len mismatch (MLP needs exact seq_len)
            if model_name == 'MLP' and data['seq_len'] != seq_len:
                continue
            
            logger.info(f"\nRun {run_idx}/{total_runs}: {model_name} | config={config}")
            
            try:
                result = train_with_config(data, model_name, config, logger)
                all_results.append(result)
                
                optimization_history.append({
                    'run': run_idx,
                    'model': model_name,
                    'config': config,
                    'r2': result['metrics']['r2'],
                    'rmse': result['metrics']['rmse'],
                    'time': result['metrics']['train_time']
                })
                
            except Exception as e:
                logger.error(f"Failed: {e}")
    
    # Analyze results
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*60)
    
    # Sort by R2
    all_results_sorted = sorted(all_results, key=lambda x: x['metrics']['r2'], reverse=True)
    
    # Leaderboard
    logger.info(f"\n{'Rank':<6}{'Model':<10}{'Config':<30}{'R2':>10}{'RMSE':>10}{'Time':>10}")
    logger.info("-"*70)
    
    for i, r in enumerate(all_results_sorted[:20], 1):  # Top 20
        config_str = f"lr={r['config']['learning_rate']}, bs={r['config']['batch_size']}, hs={r['config']['hidden_size']}, sl={r['config']['seq_len']}"
        m = r['metrics']
        logger.info(f"{i:<6}{r['model']:<10}{config_str:<30}{m['r2']:>10.4f}{m['rmse']:>10.4f}{m['train_time']:>9.1f}s")
    
    # Best config per model
    logger.info("\n" + "-"*60)
    logger.info("BEST CONFIG PER MODEL")
    logger.info("-"*60)
    
    best_per_model = {}
    for model_name in models:
        model_results = [r for r in all_results if r['model'] == model_name]
        if model_results:
            best = max(model_results, key=lambda x: x['metrics']['r2'])
            best_per_model[model_name] = best
            logger.info(f"\n[{model_name}] Best R2={best['metrics']['r2']:.4f}")
            logger.info(f"  Config: {best['config']}")
            logger.info(f"  RMSE: {best['metrics']['rmse']:.4f}")
            logger.info(f"  Time: {best['metrics']['train_time']:.1f}s")
    
    # Overall best
    if all_results_sorted:
        overall_best = all_results_sorted[0]
        logger.info("\n" + "-"*60)
        logger.info(f"[OVERALL BEST] {overall_best['model']}")
        logger.info(f"  R2: {overall_best['metrics']['r2']:.4f}")
        logger.info(f"  Config: {overall_best['config']}")
    
    return {
        'all_results': all_results_sorted,
        'best_per_model': best_per_model,
        'optimization_history': optimization_history,
        'search_space': search_space,
        'total_runs': len(all_results),
        'timestamp': datetime.now().isoformat()
    }


def save_optimization_results(results: Dict, output_dir: Path):
    """Save optimization results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full results
    with open(output_dir / f"optimization_results_{ts}.json", 'w', encoding='utf-8') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            return obj
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    
    # History CSV
    with open(output_dir / f"optimization_history_{ts}.csv", 'w', encoding='utf-8') as f:
        f.write("run,model,lr,batch_size,hidden_size,seq_len,epochs,r2,rmse,time\n")
        for h in results['optimization_history']:
            c = h['config']
            f.write(f"{h['run']},{h['model']},{c['learning_rate']},{c['batch_size']},{c['hidden_size']},{c['seq_len']},{c['epochs']},{h['r2']:.6f},{h['rmse']:.6f},{h['time']:.1f}\n")
    
    # Best configs summary
    with open(output_dir / f"best_configs_{ts}.json", 'w', encoding='utf-8') as f:
        best_summary = {}
        for model, best in results['best_per_model'].items():
            best_summary[model] = {
                'config': best['config'],
                'r2': best['metrics']['r2'],
                'rmse': best['metrics']['rmse'],
                'train_time': best['metrics']['train_time']
            }
        json.dump(best_summary, f, indent=2)
    
    print(f"\n[DONE] Results saved to {output_dir}")
    print(f"  - optimization_results_{ts}.json")
    print(f"  - optimization_history_{ts}.csv")
    print(f"  - best_configs_{ts}.json")


# ============================================================
# Main
# ============================================================

def main():
    # Setup
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"optimization_{ts}.log"
    logger = setup_logger('optimizer', str(log_file))
    
    # Data path
    data_path = str(project_root / "dataset/Location1.csv")
    
    # Custom search space (smaller for faster demo)
    search_space = {
        'learning_rate': [0.001, 0.0005],
        'batch_size': [32, 64],
        'hidden_size': [32, 64],
        'seq_len': [12, 24],
        'epochs': [10, 20],
    }
    
    # Run optimization
    results = run_optimization(
        data_path=data_path,
        models=['MLP', 'LSTM', 'GRU'],
        search_space=search_space,
        max_configs=20,
        logger=logger
    )
    
    # Save results
    output_dir = project_root / "data/results"
    save_optimization_results(results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Total runs: {results['total_runs']}")
    print(f"Log file: {log_file}")
    
    if results['all_results']:
        best = results['all_results'][0]
        print(f"\n[Best Model] {best['model']}")
        print(f"  R2: {best['metrics']['r2']:.4f}")
        print(f"  Config: lr={best['config']['learning_rate']}, batch_size={best['config']['batch_size']}, hidden_size={best['config']['hidden_size']}")


if __name__ == "__main__":
    main()