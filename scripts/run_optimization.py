"""
Run Auto Optimization - Single Run
"""

import sys
import os
from pathlib import Path
import importlib.util

# Set working directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Load module directly (avoiding global scripts package conflict)
spec = importlib.util.spec_from_file_location(
    "auto_optimizer", 
    project_root / "scripts" / "auto_optimizer.py"
)
auto_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_optimizer)

AutoOptimizer = auto_optimizer.AutoOptimizer

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Auto Optimization")
    parser.add_argument("--models", type=str, default="gradient_boosting", 
                        help="Models to optimize (comma-separated)")
    parser.add_argument("--trials", type=int, default=5, help="Trials per model")
    parser.add_argument("--target-r2", type=float, default=0.99, help="Target R2")
    args = parser.parse_args()
    
    print("=" * 70)
    print("AUTO OPTIMIZATION")
    print("=" * 70)
    print(f"Models: {args.models}")
    print(f"Trials per model: {args.trials}")
    print(f"Target R2: {args.target_r2}")
    
    optimizer = AutoOptimizer()
    models = [m.strip() for m in args.models.split(",")]
    
    results = []
    for model_name in models:
        result = optimizer.optimize_model(model_name, n_trials=args.trials)
        if result:
            results.append(result)
            optimizer.history.add_entry(result)
            
            if result['metrics']['r2'] >= args.target_r2:
                print(f"\n[TARGET REACHED] R2={result['metrics']['r2']:.4f}")
    
    if results:
        best = max(results, key=lambda x: x['metrics']['r2'])
        
        print("\n" + "=" * 70)
        print("BEST RESULT")
        print("=" * 70)
        print(f"Model: {best['model']}")
        print(f"R2: {best['metrics']['r2']:.4f}")
        print(f"RMSE: {best['metrics']['rmse']:.4f}")
        print(f"MAE: {best['metrics']['mae']:.4f}")
        print(f"Params: {best['params']}")
        
        # Auto commit if good result
        if best['metrics']['r2'] > 0.98:
            print("\n[INFO] Good result! Attempting to commit...")
            commit_msg = f"Auto-optimize: {best['model']} R2={best['metrics']['r2']:.4f}"
            if optimizer.git.commit(commit_msg):
                print(f"  Committed: {commit_msg}")
                optimizer.git.push()


if __name__ == "__main__":
    main()
