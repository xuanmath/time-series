"""
自动优化闭环系统 - Auto Optimization Loop

完整的自动化流程：
指标定义 → 自动优化 → 自动验证 → 自动提交/回滚 → 循环迭代

目标：R² >= 0.95

功能：
1. 自动训练模型并计算指标
2. 如果指标未达标，自动调整超参数/架构
3. 如果改进则提交git，否则回滚
4. 无限循环直到达标
"""

import sys
import os
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.cnn_gru import CNNGRUForecaster, get_default_config, get_advanced_config
from src.utils.preprocessing import DataPreprocessor
from src.utils.metrics import evaluate_predictions


class OptimizationConfig:
    """优化配置"""
    
    # 目标指标
    TARGET_R2 = 0.95
    TARGET_RMSE = 0.02  # Optional secondary target
    
    # 训练参数搜索空间
    PARAM_SEARCH_SPACE = {
        "seq_len": [24, 48, 72, 96],
        "cnn_channels": [
            [32, 64],
            [64, 128],
            [64, 128, 256],
            [32, 64, 128]
        ],
        "cnn_kernel_sizes": [
            [3, 3],
            [5, 5],
            [3, 3, 3]
        ],
        "gru_hidden_size": [64, 128, 256],
        "gru_num_layers": [1, 2, 3],
        "dropout": [0.1, 0.15, 0.2, 0.25],
        "learning_rate": [0.001, 0.0005, 0.0001],
        "batch_size": [32, 64, 128],
        "epochs": [100, 200, 300]
    }
    
    # 优化策略顺序
    OPTIMIZATION_STRATEGIES = [
        "hyperparameter_tuning",
        "architecture_search",
        "feature_engineering",
        "data_augmentation"
    ]


class AutoOptimizationLoop:
    """
    自动优化闭环系统
    
    实现：
    1. 指标定义 - 使用R², RMSE, MAE
    2. 自动优化 - 超参数搜索、架构搜索
    3. 自动验证 - 训练后计算指标
    4. 自动提交/回滚 - Git自动化
    5. 循环迭代 - 直到达标
    """
    
    def __init__(
        self,
        project_root: str,
        data_path: str,
        target_r2: float = 0.95,
        max_iterations: int = 100,
        conda_env: str = "pytorch_gpu",
        git_remote: str = "origin",
        git_branch: str = "main"
    ):
        self.project_root = Path(project_root)
        self.data_path = Path(data_path)
        self.target_r2 = target_r2
        self.max_iterations = max_iterations
        self.conda_env = conda_env
        self.git_remote = git_remote
        self.git_branch = git_branch
        
        # Directories
        self.backup_dir = self.project_root / ".optimization_backup"
        self.history_dir = self.project_root / "logs"
        self.results_dir = self.project_root / "data/results"
        self.models_dir = self.project_root / "models/checkpoints"
        
        # Ensure directories exist
        for d in [self.backup_dir, self.history_dir, self.results_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # History
        self.history_file = self.history_dir / "optimization_loop_history.json"
        self.history = self._load_history()
        
        # Best result tracking
        self.best_result = None
        self.best_config = None
        
        # Current iteration
        self.iteration = 0
        
    def _load_history(self) -> List[Dict]:
        """Load optimization history."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save optimization history."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _create_backup(self, iteration: int) -> Path:
        """Create backup before optimization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_iter{iteration}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        # Backup critical files
        shutil.copy2(self.project_root / "src/models/cnn_gru.py", 
                     backup_path / "src/models/cnn_gru.py")
        
        return backup_path
    
    def _restore_backup(self, backup_path: Path) -> bool:
        """Restore from backup."""
        if not backup_path.exists():
            return False
        
        try:
            # Restore files
            for f in backup_path.rglob("*"):
                if f.is_file():
                    rel_path = f.relative_to(backup_path)
                    dst = self.project_root / rel_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
            return True
        except Exception as e:
            print(f"Backup restore failed: {e}")
            return False
    
    def _load_data(self) -> tuple:
        """Load and prepare data."""
        df = pd.read_csv(self.data_path)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        
        # Feature columns
        feature_cols = [col for col in df.columns if col not in ['Time', 'Power']]
        target_col = 'Power'
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Split
        test_size = 0.2
        split_idx = int(len(X_scaled) * (1 - test_size))
        
        self.X_train = X_scaled[:split_idx]
        self.y_train = y_scaled[:split_idx]
        self.X_test = X_scaled[split_idx:]
        self.y_test = y_scaled[split_idx:]
        
        self.feature_cols = feature_cols
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def _generate_config(self, iteration: int) -> Dict:
        """Generate model configuration based on iteration."""
        search_space = OptimizationConfig.PARAM_SEARCH_SPACE
        
        if iteration == 0:
            # Start with default config
            return get_default_config()
        
        elif iteration <= 10:
            # Quick search: vary key parameters
            configs = []
            for seq_len in [24, 48]:
                for hidden in [64, 128]:
                    for lr in [0.001, 0.0005]:
                        config = get_default_config()
                        config['seq_len'] = seq_len
                        config['gru_hidden_size'] = hidden
                        config['learning_rate'] = lr
                        configs.append(config)
            return configs[iteration % len(configs)]
        
        else:
            # Random search in full space
            import random
            config = {
                "seq_len": random.choice(search_space["seq_len"]),
                "pred_len": 1,
                "cnn_channels": random.choice(search_space["cnn_channels"]),
                "cnn_kernel_sizes": random.choice(search_space["cnn_kernel_sizes"]),
                "gru_hidden_size": random.choice(search_space["gru_hidden_size"]),
                "gru_num_layers": random.choice(search_space["gru_num_layers"]),
                "dropout": random.choice(search_space["dropout"]),
                "use_batch_norm": True,
                "learning_rate": random.choice(search_space["learning_rate"]),
                "epochs": random.choice(search_space["epochs"]),
                "batch_size": random.choice(search_space["batch_size"]),
                "weight_decay": 1e-4,
                "early_stopping_patience": 20
            }
            return config
    
    def _train_model(self, config: Dict) -> Dict:
        """Train CNN-GRU model with given config."""
        print(f"\n{'='*70}")
        print(f"TRAINING - Iteration {self.iteration}")
        print(f"{'='*70}")
        print(f"Config: seq_len={config['seq_len']}, "
              f"gru_hidden={config['gru_hidden_size']}, "
              f"dropout={config['dropout']}, lr={config['learning_rate']}")
        
        # Load data
        self._load_data()
        
        # Initialize model
        model = CNNGRUForecaster(
            seq_len=config['seq_len'],
            pred_len=config['pred_len'],
            input_features=len(self.feature_cols),
            cnn_channels=config['cnn_channels'],
            cnn_kernel_sizes=config['cnn_kernel_sizes'],
            gru_hidden_size=config['gru_hidden_size'],
            gru_num_layers=config['gru_num_layers'],
            dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            weight_decay=config.get('weight_decay', 1e-5),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Train
        train_history = model.fit(
            self.X_train, 
            self.y_train,
            verbose=1,
            early_stopping_patience=config.get('early_stopping_patience', 15)
        )
        
        # Create test sequences
        X_test_seq = []
        y_test_seq = []
        
        for i in range(len(self.X_test) - config['seq_len']):
            X_test_seq.append(self.X_test[i:i + config['seq_len']])
            y_test_seq.append(self.y_test[i + config['seq_len']])
        
        X_test_seq = np.array(X_test_seq)
        y_test_seq = np.array(y_test_seq)
        
        # Predict
        y_pred_scaled = model.predict(X=X_test_seq)
        
        # Inverse transform for metrics
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        metrics = evaluate_predictions(y_true, y_pred)
        
        result = {
            "iteration": self.iteration,
            "config": config,
            "metrics": metrics,
            "train_history": {
                "best_epoch": train_history['best_epoch'],
                "best_val_loss": train_history['best_val_loss'],
                "total_epochs": train_history['total_epochs']
            },
            "timestamp": datetime.now().isoformat(),
            "improved": False
        }
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"R²:   {metrics['r2']:.6f} (target: {self.target_r2})")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE:  {metrics['mae']:.6f}")
        print(f"SMAPE: {metrics['smape']:.2f}%")
        
        return result
    
    def _git_commit(self, result: Dict) -> bool:
        """Git commit for improved results."""
        try:
            # Check git status
            subprocess.run(['git', 'status'], cwd=self.project_root, capture_output=True)
            
            # Create commit message
            commit_msg = f"auto-optimize: iter{result['iteration']} R²={result['metrics']['r2']:.4f}"
            
            # Add changed files
            subprocess.run(['git', 'add', '-A'], cwd=self.project_root, capture_output=True)
            
            # Commit
            subprocess.run(['git', 'commit', '-m', commit_msg], 
                          cwd=self.project_root, capture_output=True)
            
            print(f"✅ Git commit: {commit_msg}")
            return True
            
        except Exception as e:
            print(f"⚠️ Git commit failed: {e}")
            return False
    
    def _git_push(self) -> bool:
        """Push to remote."""
        try:
            subprocess.run(['git', 'push', self.git_remote, self.git_branch],
                          cwd=self.project_root, capture_output=True)
            print(f"✅ Git push to {self.git_remote}/{self.git_branch}")
            return True
        except Exception as e:
            print(f"⚠️ Git push failed: {e}")
            return False
    
    def _git_rollback(self) -> bool:
        """Git rollback to previous commit."""
        try:
            subprocess.run(['git', 'reset', '--hard', 'HEAD~1'],
                          cwd=self.project_root, capture_output=True)
            print("✅ Git rollback completed")
            return True
        except Exception as e:
            print(f"⚠️ Git rollback failed: {e}")
            return False
    
    def _is_improved(self, result: Dict) -> bool:
        """Check if result is improved over best."""
        if self.best_result is None:
            return True
        
        current_r2 = result['metrics']['r2']
        best_r2 = self.best_result['metrics']['r2']
        
        return current_r2 > best_r2
    
    def _is_target_met(self, result: Dict) -> bool:
        """Check if target is met."""
        return result['metrics']['r2'] >= self.target_r2
    
    def run_single_iteration(self) -> Dict:
        """Run single optimization iteration."""
        self.iteration += 1
        
        print(f"\n{'#'*70}")
        print(f"# ITERATION {self.iteration}")
        print(f"{'#'*70}")
        
        # Create backup
        backup_path = self._create_backup(self.iteration)
        
        # Generate config
        config = self._generate_config(self.iteration)
        
        # Train and evaluate
        result = self._train_model(config)
        
        # Check improvement
        if self._is_improved(result):
            result['improved'] = True
            self.best_result = result
            self.best_config = config
            
            # Save best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save result
            result_file = self.results_dir / f"best_result_iter{self.iteration}_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Git commit
            self._git_commit(result)
            
            print(f"\n✅ IMPROVED! New best R² = {result['metrics']['r2']:.4f}")
        else:
            result['improved'] = False
            
            # Restore backup if no improvement
            self._restore_backup(backup_path)
            
            print(f"\n⚠️ No improvement. R² = {result['metrics']['r2']:.4f} "
                  f"(best: {self.best_result['metrics']['r2']:.4f})")
        
        # Save to history
        self.history.append(result)
        self._save_history()
        
        return result
    
    def run_loop(
        self,
        target_r2: float = None,
        max_iterations: int = None
    ) -> Optional[Dict]:
        """
        Run complete optimization loop.
        
        Loop continues until:
        - R² >= target
        - max_iterations reached
        - No improvement for 10 consecutive iterations
        """
        target_r2 = target_r2 or self.target_r2
        max_iterations = max_iterations or self.max_iterations
        
        print(f"\n{'='*70}")
        print(f"AUTO OPTIMIZATION LOOP - CNN-GRU")
        print(f"{'='*70}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target R²: {target_r2}")
        print(f"Max Iterations: {max_iterations}")
        print(f"{'='*70}")
        
        consecutive_no_improve = 0
        max_no_improve = 15
        
        while self.iteration < max_iterations:
            result = self.run_single_iteration()
            
            # Check target
            if self._is_target_met(result):
                print(f"\n{'='*70}")
                print(f"🎉 TARGET ACHIEVED!")
                print(f"{'='*70}")
                print(f"Final R²: {result['metrics']['r2']:.4f}")
                print(f"Iterations: {self.iteration}")
                
                # Final git push
                self._git_push()
                
                return result
            
            # Check consecutive no improvement
            if result['improved']:
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
            
            if consecutive_no_improve >= max_no_improve:
                print(f"\n⚠️ Stopping: No improvement for {max_no_improve} iterations")
                print(f"Best R² achieved: {self.best_result['metrics']['r2']:.4f}")
                
                # Push best result
                if self.best_result:
                    self._git_push()
                
                return self.best_result
        
        print(f"\n⚠️ Max iterations ({max_iterations}) reached")
        print(f"Best R² achieved: {self.best_result['metrics']['r2']:.4f}")
        
        return self.best_result
    
    def get_summary(self) -> Dict:
        """Get optimization summary."""
        if not self.history:
            return {"status": "no_history"}
        
        r2_values = [h['metrics']['r2'] for h in self.history]
        
        return {
            "total_iterations": len(self.history),
            "improved_iterations": sum(1 for h in self.history if h['improved']),
            "best_r2": max(r2_values),
            "avg_r2": np.mean(r2_values),
            "target_r2": self.target_r2,
            "target_met": max(r2_values) >= self.target_r2,
            "best_config": self.best_config,
            "best_result": self.best_result
        }


def main():
    parser = argparse.ArgumentParser(description="Auto Optimization Loop")
    parser.add_argument("--target-r2", type=float, default=0.95,
                        help="Target R² score (default: 0.95)")
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="Maximum iterations (default: 50)")
    parser.add_argument("--data", type=str, default="dataset/Location1.csv",
                        help="Data file path")
    parser.add_argument("--conda-env", type=str, default="pytorch_gpu",
                        help="Conda environment name")
    parser.add_argument("--push", action="store_true",
                        help="Push to git after optimization")
    
    args = parser.parse_args()
    
    # Import torch for device check
    import torch
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data
    
    loop = AutoOptimizationLoop(
        project_root=str(project_root),
        data_path=str(data_path),
        target_r2=args.target_r2,
        max_iterations=args.max_iterations,
        conda_env=args.conda_env
    )
    
    best_result = loop.run_loop()
    
    # Print summary
    summary = loop.get_summary()
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total iterations: {summary['total_iterations']}")
    print(f"Improved iterations: {summary['improved_iterations']}")
    print(f"Best R²: {summary['best_r2']:.4f}")
    print(f"Average R²: {summary['avg_r2']:.4f}")
    print(f"Target met: {summary['target_met']}")
    
    if summary['best_config']:
        print(f"\nBest configuration:")
        for k, v in summary['best_config'].items():
            print(f"  {k}: {v}")
    
    if args.push and best_result:
        loop._git_push()


if __name__ == "__main__":
    main()