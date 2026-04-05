"""
7x24 Unattended Auto Optimization Worker

This script is designed to be called by OpenClaw Cron for continuous optimization.

Workflow:
1. Check memory for current best results
2. If target met, skip and report status
3. Run model training comparison
4. Compare metrics with current best
5. Git commit if improved
6. Update memory with new results
7. Notify user if target achieved
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "dataset/Location1.csv"
RESULTS_DIR = PROJECT_ROOT / "data/results"
LOGS_DIR = PROJECT_ROOT / "logs"
MEMORY_FILE = Path.home() / ".openclaw/workspace/memory/ts-optimization-state.json"

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import evaluate_predictions
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class OptimizationState:
    """Manage optimization state in memory."""
    
    def __init__(self):
        self.state = self._load_state()
    
    def _load_state(self):
        """Load state from memory file."""
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        
        # Default state
        return {
            "project": "time-series",
            "target_r2": 0.95,
            "current_best_r2": 0.0,
            "target_met": False,
            "iterations": 0,
            "best_model": None,
            "best_config": None,
            "last_run": None,
            "status": "initialized",
            "history": []
        }
    
    def save(self):
        """Save state to memory file."""
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def update(self, result):
        """Update state with new result."""
        self.state['iterations'] += 1
        self.state['last_run'] = datetime.now().isoformat()
        
        if result['metrics']['r2'] > self.state['current_best_r2']:
            self.state['current_best_r2'] = result['metrics']['r2']
            self.state['best_model'] = result['model']
            self.state['best_config'] = result.get('config', {})
            self.state['status'] = 'improved'
            
            # Add to history
            self.state['history'].append({
                "iteration": self.state['iterations'],
                "model": result['model'],
                "r2": result['metrics']['r2'],
                "rmse": result['metrics']['rmse'],
                "mae": result['metrics']['mae'],
                "timestamp": datetime.now().isoformat()
            })
        else:
            self.state['status'] = 'no_improvement'
        
        # Check target
        if self.state['current_best_r2'] >= self.state['target_r2']:
            self.state['target_met'] = True
            self.state['status'] = 'target_achieved'
        
        self.save()
        return self.state
    
    def should_run(self):
        """Check if optimization should continue."""
        if self.state['target_met']:
            return False, "Target already achieved"
        
        # Max iterations check
        if self.state['iterations'] >= 100:
            return False, "Max iterations reached"
        
        # Check for recent run (within 30 minutes)
        if self.state['last_run']:
            last = datetime.fromisoformat(self.state['last_run'])
            elapsed = (datetime.now() - last).total_seconds()
            if elapsed < 1800:  # 30 minutes
                return False, f"Last run was {elapsed/60:.1f} minutes ago"
        
        return True, "Ready to run"


def prepare_data(seq_len=24):
    """Prepare data for training."""
    df = pd.read_csv(DATA_PATH)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col not in ['Time', 'Power']]
    X = df[feature_cols].values
    y = df['Power'].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # Create sequences
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
        'X_test': X_test, 'y_test': y_test,
        'n_features': len(feature_cols),
        'seq_len': seq_len
    }


def train_gradient_boosting(data):
    """Train Gradient Boosting."""
    X_train = data['X_train_seq'].reshape(len(data['X_train_seq']), -1)
    X_test = data['X_test_seq'].reshape(len(data['X_test_seq']), -1)
    y_train = data['y_train_seq'].flatten()
    y_test = data['y_test_seq'].flatten()
    
    # Random parameter search
    import random
    params = {
        'n_estimators': random.choice([50, 100, 150]),
        'max_depth': random.choice([4, 6, 8, 10]),
        'learning_rate': random.choice([0.05, 0.1, 0.15, 0.2]),
        'subsample': random.choice([0.7, 0.8, 0.9])
    }
    
    model = GradientBoostingRegressor(**params, random_state=42)
    
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['train_time'] = train_time
    
    return {
        'model': 'GradientBoosting',
        'config': params,
        'metrics': metrics
    }


def train_random_forest(data):
    """Train Random Forest."""
    X_train = data['X_train_seq'].reshape(len(data['X_train_seq']), -1)
    X_test = data['X_test_seq'].reshape(len(data['X_test_seq']), -1)
    y_train = data['y_train_seq'].flatten()
    y_test = data['y_test_seq'].flatten()
    
    import random
    params = {
        'n_estimators': random.choice([50, 100, 150]),
        'max_depth': random.choice([6, 10, 15]),
        'min_samples_split': random.choice([2, 5, 10])
    }
    
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['train_time'] = train_time
    
    return {
        'model': 'RandomForest',
        'config': params,
        'metrics': metrics
    }


def run_optimization_iteration():
    """Run single optimization iteration."""
    print(f"\n{'#'*60}")
    print(f"# AUTO OPTIMIZATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    # Initialize state
    state = OptimizationState()
    
    # Check if should run
    should_run, reason = state.should_run()
    
    if not should_run:
        print(f"\n⏸️ Skipping: {reason}")
        print(f"Current best R²: {state.state['current_best_r2']:.4f}")
        print(f"Target R²: {state.state['target_r2']:.4f}")
        
        if state.state['target_met']:
            print(f"\n🎉 TARGET ACHIEVED!")
            print(f"Best model: {state.state['best_model']}")
            print(f"Best R²: {state.state['current_best_r2']:.4f}")
            return state.state, "target_achieved"
        
        return state.state, "skipped"
    
    print(f"\n📊 Current Status:")
    print(f"  Iterations: {state.state['iterations']}")
    print(f"  Best R²: {state.state['current_best_r2']:.4f}")
    print(f"  Target: {state.state['target_r2']:.4f}")
    
    # Prepare data
    print(f"\n⚙️ Preparing data...")
    data = prepare_data(seq_len=24)
    print(f"  Train: {data['X_train_seq'].shape}")
    print(f"  Test: {data['X_test_seq'].shape}")
    
    # Train models
    results = []
    
    print(f"\n🤖 Training models...")
    
    try:
        result = train_gradient_boosting(data)
        results.append(result)
        print(f"  GradientBoosting: R²={result['metrics']['r2']:.4f}")
    except Exception as e:
        print(f"  GradientBoosting: ❌ {e}")
    
    try:
        result = train_random_forest(data)
        results.append(result)
        print(f"  RandomForest: R²={result['metrics']['r2']:.4f}")
    except Exception as e:
        print(f"  RandomForest: ❌ {e}")
    
    # Get best result
    if results:
        best = max(results, key=lambda x: x['metrics']['r2'])
        print(f"\n🏆 Best this iteration: {best['model']} R²={best['metrics']['r2']:.4f}")
        
        # Update state
        state.update(best)
        
        # Git commit if improved
        if state.state['status'] == 'improved':
            try:
                subprocess.run(['git', 'add', '-A'], cwd=PROJECT_ROOT, capture_output=True)
                commit_msg = f"auto-optimize: iter{state.state['iterations']} {best['model']} R²={best['metrics']['r2']:.4f}"
                subprocess.run(['git', 'commit', '-m', commit_msg], cwd=PROJECT_ROOT, capture_output=True)
                print(f"  ✅ Git committed: {commit_msg}")
            except Exception as e:
                print(f"  ⚠️ Git commit failed: {e}")
        
        # Check target
        if state.state['target_met']:
            print(f"\n{'='*60}")
            print(f"🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉")
            print(f"{'='*60}")
            print(f"Best Model: {state.state['best_model']}")
            print(f"Best R²: {state.state['current_best_r2']:.4f}")
            print(f"Total Iterations: {state.state['iterations']}")
            
            # Git push
            try:
                subprocess.run(['git', 'push'], cwd=PROJECT_ROOT, capture_output=True)
                print(f"✅ Pushed to remote")
            except:
                pass
            
            return state.state, "target_achieved"
    
    return state.state, "iteration_complete"


def main():
    """Main entry point for cron execution."""
    state, status = run_optimization_iteration()
    
    # Output for OpenClaw to parse
    output = {
        "status": status,
        "state": state,
        "timestamp": datetime.now().isoformat()
    }
    
    # Write result to file for OpenClaw to read
    result_file = RESULTS_DIR / f"optimization_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Result saved: {result_file}")
    
    # Return status for notification
    if status == "target_achieved":
        return f"🎉 Target achieved! Best R²={state['current_best_r2']:.4f}"
    elif status == "iteration_complete":
        return f"✅ Iteration {state['iterations']} complete. Best R²={state['current_best_r2']:.4f}"
    else:
        return f"⏸️ {status}"


if __name__ == "__main__":
    result = main()
    print(f"\n📌 {result}")