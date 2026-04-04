"""
AutoML Optimizer for Time Series Forecasting
Implements: Metrics Definition → Auto Optimization → Auto Validation → Auto Commit/Rollback → Iteration
"""

import json
import yaml
import random
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class MetricsCalculator:
    """Calculate evaluation metrics."""
    
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # SMAPE
        smape = float(100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'smape': smape
        }
    
    @staticmethod
    def composite_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate composite score from metrics."""
        # Normalize metrics (higher is better for composite)
        r2_score = metrics['r2']  # Already 0-1, higher is better
        rmse_score = 1 / (1 + metrics['rmse'])  # Invert, higher is better
        mae_score = 1 / (1 + metrics['mae'])  # Invert, higher is better
        smape_score = 1 / (1 + metrics['smape'] / 100)  # Invert and normalize
        
        return (
            weights.get('r2', 0.5) * r2_score +
            weights.get('rmse', 0.25) * rmse_score +
            weights.get('mae', 0.15) * mae_score +
            weights.get('smape', 0.10) * smape_score
        )


class HyperparameterSampler:
    """Sample hyperparameters from search space."""
    
    def __init__(self, search_space: Dict[str, List]):
        self.search_space = search_space
    
    def random_sample(self) -> Dict[str, Any]:
        """Random sampling from search space."""
        return {
            key: random.choice(values) if isinstance(values, list) else values
            for key, values in self.search_space.items()
        }
    
    def grid_search(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        import itertools
        
        keys = list(self.search_space.keys())
        values = [self.search_space[k] if isinstance(self.search_space[k], list) else [self.search_space[k]] 
                  for k in keys]
        
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]


class ModelTrainer:
    """Train and evaluate models."""
    
    def __init__(self, data_path: str, config: Dict):
        self.data_path = data_path
        self.config = config
        self.device = None
        
    def load_data(self, seq_len: int = 24):
        """Load and preprocess data."""
        import torch
        
        df = pd.read_csv(self.data_path)
        data = df['Power'].values.reshape(-1, 1)
        
        # Min-max scaling
        self.data_min = data.min()
        self.data_max = data.max()
        data_scaled = (data - self.data_min) / (self.data_max - self.data_min)
        
        # Split
        train_size = int(len(data_scaled) * 0.8)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        
        # Create sequences
        X_train, y_train = self._create_sequences(train_data, seq_len)
        X_test, y_test = self._create_sequences(test_data, seq_len)
        
        # To tensors
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).to(self.device)
        
        return X_train_t, y_train_t, X_test_t, y_test_t, test_data
    
    def _create_sequences(self, data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)
    
    def train_sklearn_model(self, model_name: str, params: Dict) -> Tuple[Any, Dict]:
        """Train sklearn model."""
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Load full data for sklearn
        df = pd.read_csv(self.data_path)
        data = df['Power'].values
        
        # Create lagged features
        seq_len = params.get('seq_len', 24)
        X, y = self._create_features(data, seq_len)
        
        # Split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create model
        if model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                random_state=42,
                verbose=0
            )
        elif model_name == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate(y_test, predictions)
        metrics['train_time'] = train_time
        
        return model, metrics
    
    def _create_features(self, data, seq_len):
        """Create lagged features for sklearn models."""
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)
    
    def train_pytorch_model(self, model_name: str, params: Dict) -> Tuple[Any, Dict]:
        """Train PyTorch model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        seq_len = params.get('seq_len', 24)
        X_train, y_train, X_test, y_test, test_data = self.load_data(seq_len)
        
        # Build model
        if model_name == 'lstm':
            model = self._build_lstm(seq_len, params)
        elif model_name == 'gru':
            model = self._build_gru(seq_len, params)
        elif model_name == 'transformer':
            model = self._build_transformer(seq_len, params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(self.device)
        
        # Train
        start_time = time.time()
        self._train_pytorch(model, X_train, y_train, params)
        train_time = time.time() - start_time
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy()
        
        # Inverse transform
        y_test_np = y_test.cpu().numpy()
        predictions_orig = predictions * (self.data_max - self.data_min) + self.data_min
        y_test_orig = y_test_np * (self.data_max - self.data_min) + self.data_min
        
        metrics = MetricsCalculator.calculate(y_test_orig, predictions_orig)
        metrics['train_time'] = train_time
        
        return model, metrics
    
    def _build_lstm(self, seq_len, params):
        import torch.nn as nn
        class LSTMNet(nn.Module):
            def __init__(self, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, 
                                   dropout=dropout if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_size, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        return LSTMNet(params.get('hidden_size', 64), params.get('num_layers', 2), params.get('dropout', 0.1))
    
    def _build_gru(self, seq_len, params):
        import torch.nn as nn
        class GRUNet(nn.Module):
            def __init__(self, hidden_size, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(1, hidden_size, num_layers, batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_size, 1)
            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])
        return GRUNet(params.get('hidden_size', 64), params.get('num_layers', 2), params.get('dropout', 0.1))
    
    def _build_transformer(self, seq_len, params):
        import torch.nn as nn
        class TransformerNet(nn.Module):
            def __init__(self, d_model, nhead, num_layers, dropout):
                super().__init__()
                self.embedding = nn.Linear(1, d_model)
                self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.fc = nn.Linear(d_model, 1)
            def forward(self, x):
                x = self.embedding(x) + self.pos_encoder
                x = self.transformer(x)
                return self.fc(x[:, -1, :])
        return TransformerNet(params.get('d_model', 64), params.get('nhead', 4), 
                              params.get('num_layers', 2), params.get('dropout', 0.1))
    
    def _train_pytorch(self, model, X_train, y_train, params):
        import torch
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('learning_rate', 1e-3))
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        epochs = params.get('epochs', 50)
        patience = params.get('patience', 10)
        best_loss = float('inf')
        patience_counter = 0
        
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
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break


class GitManager:
    """Manage Git operations for auto-commit."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
    
    def run_git(self, command: str) -> Tuple[int, str, str]:
        """Run a git command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Timeout"
    
    def has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        code, stdout, _ = self.run_git("git status --short")
        return len(stdout.strip()) > 0
    
    def commit(self, message: str) -> bool:
        """Commit changes."""
        if not self.has_changes():
            return False
        
        self.run_git("git add .")
        code, _, stderr = self.run_git(f'git commit -m "{message}"')
        return code == 0
    
    def push(self) -> bool:
        """Push to remote."""
        code, _, stderr = self.run_git("git push origin main")
        return code == 0
    
    def get_current_commit(self) -> str:
        """Get current commit hash."""
        _, stdout, _ = self.run_git("git rev-parse --short HEAD")
        return stdout.strip()


class OptimizationHistory:
    """Track optimization history."""
    
    def __init__(self, history_path: str):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load()
    
    def _load(self) -> List[Dict]:
        """Load history from file."""
        if self.history_path.exists():
            with open(self.history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save(self):
        """Save history to file."""
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def add_entry(self, entry: Dict):
        """Add an entry to history."""
        entry['timestamp'] = datetime.now().isoformat()
        self.history.append(entry)
        self.save()
    
    def get_best(self, metric: str = 'r2') -> Optional[Dict]:
        """Get best result by metric."""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x.get('metrics', {}).get(metric, -float('inf')))
    
    def get_improvement_trend(self, metric: str = 'r2', n: int = 10) -> List[float]:
        """Get improvement trend."""
        recent = self.history[-n:]
        return [e.get('metrics', {}).get(metric, 0) for e in recent]


class AutoOptimizer:
    """Main auto-optimization class."""
    
    def __init__(self, config_path: str = "configs/metrics.yaml"):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.trainer = ModelTrainer("data/raw/wind_power.csv", self.config)
        self.git = GitManager(PROJECT_ROOT)
        self.history = OptimizationHistory("logs/optimization_history.json")
        
        # Metrics config
        self.primary_metric = self.config['primary_metric']
        self.targets = self.config['targets']
        self.weights = self.config['weights']
    
    def is_target_reached(self, metrics: Dict) -> bool:
        """Check if target metrics are reached."""
        for metric, target in self.targets.items():
            if metric in metrics:
                if metric == 'r2':
                    if metrics[metric] < target:
                        return False
                else:
                    if metrics[metric] > target:
                        return False
        return True
    
    def is_improvement(self, metrics: Dict) -> bool:
        """Check if this is an improvement over best."""
        best = self.history.get_best(self.primary_metric)
        if best is None:
            return True
        
        new_score = metrics.get(self.primary_metric, 0)
        best_score = best.get('metrics', {}).get(self.primary_metric, 0)
        
        if self.config['optimization_direction'] == 'maximize':
            return new_score > best_score
        else:
            return new_score < best_score
    
    def optimize_model(self, model_name: str, n_trials: int = 5) -> Optional[Dict]:
        """Optimize a single model."""
        print(f"\n{'='*60}")
        print(f"Optimizing: {model_name.upper()}")
        print("=" * 60)
        
        # Get search space
        search_space = self.config['hyperparameters'].get(model_name, {})
        if not search_space:
            print(f"  No search space for {model_name}")
            return None
        
        sampler = HyperparameterSampler(search_space)
        best_result = None
        
        for trial in range(n_trials):
            # Sample hyperparameters
            params = sampler.random_sample()
            params['seq_len'] = random.choice(self.config['feature_engineering']['seq_len_options'])
            
            print(f"\n  Trial {trial+1}/{n_trials}")
            print(f"  Params: {params}")
            
            try:
                # Train model
                if model_name in ['gradient_boosting', 'random_forest']:
                    model, metrics = self.trainer.train_sklearn_model(model_name, params)
                else:
                    model, metrics = self.trainer.train_pytorch_model(model_name, params)
                
                print(f"  Metrics: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
                
                # Check if improvement
                if best_result is None or metrics['r2'] > best_result['metrics']['r2']:
                    best_result = {
                        'model': model_name,
                        'params': params,
                        'metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    print(f"  * New best for {model_name}!")
            
            except Exception as e:
                print(f"  Error: {str(e)[:50]}")
                continue
        
        return best_result
    
    def run_iteration(self, models: List[str] = None) -> Dict:
        """Run one optimization iteration."""
        print("\n" + "=" * 70)
        print(f"OPTIMIZATION ITERATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        if models is None:
            models = [m['name'] for m in self.config['models'] if m.get('enabled', True)]
        
        results = []
        for model_name in models:
            result = self.optimize_model(model_name, n_trials=3)
            if result:
                results.append(result)
        
        if not results:
            return None
        
        # Find best overall
        best = max(results, key=lambda x: x['metrics']['r2'])
        
        # Record in history
        self.history.add_entry(best)
        
        # Check if target reached
        if self.is_target_reached(best['metrics']):
            print(f"\n[TARGET REACHED] R2={best['metrics']['r2']:.4f}")
        
        # Check if improvement
        if self.is_improvement(best['metrics']):
            print(f"\n[IMPROVEMENT] New best R2={best['metrics']['r2']:.4f}")
            
            # Auto commit
            if self.config['git']['auto_commit']:
                commit_msg = f"Auto-optimize: {best['model']} R2={best['metrics']['r2']:.4f}"
                if self.git.commit(commit_msg):
                    print(f"  Committed: {commit_msg}")
                    
                    if self.config['git']['auto_push']:
                        if self.git.push():
                            print("  Pushed to GitHub")
        
        return best
    
    def run_loop(self, max_iterations: int = 50, target_r2: float = 0.99):
        """Run optimization loop until target is reached."""
        print("\n" + "=" * 70)
        print("AUTO OPTIMIZATION LOOP")
        print("=" * 70)
        print(f"Target R2: {target_r2}")
        print(f"Max Iterations: {max_iterations}")
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print("=" * 70)
            
            result = self.run_iteration()
            
            if result and result['metrics']['r2'] >= target_r2:
                print(f"\n{'='*70}")
                print("TARGET REACHED!")
                print(f"Best Model: {result['model']}")
                print(f"R2: {result['metrics']['r2']:.4f}")
                print(f"RMSE: {result['metrics']['rmse']:.4f}")
                print("=" * 70)
                return result
            
            # Show progress
            best = self.history.get_best()
            if best:
                print(f"\nCurrent Best: {best['model']} R2={best['metrics']['r2']:.4f}")
        
        print("\n[INFO] Max iterations reached")
        return self.history.get_best()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoML Optimizer")
    parser.add_argument("--loop", action="store_true", help="Run optimization loop")
    parser.add_argument("--iterations", type=int, default=20, help="Max iterations")
    parser.add_argument("--target-r2", type=float, default=0.99, help="Target R2")
    parser.add_argument("--models", type=str, default="gradient_boosting,lstm,gru", help="Models to optimize")
    args = parser.parse_args()
    
    optimizer = AutoOptimizer()
    
    if args.loop:
        optimizer.run_loop(max_iterations=args.iterations, target_r2=args.target_r2)
    else:
        models = [m.strip() for m in args.models.split(",")]
        result = optimizer.run_iteration(models)
        
        if result:
            print(f"\n{'='*60}")
            print("BEST RESULT")
            print("=" * 60)
            print(f"Model: {result['model']}")
            print(f"Params: {result['params']}")
            print(f"Metrics: R2={result['metrics']['r2']:.4f}, RMSE={result['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    main()
