"""
Run Continuous Optimization Loop
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.auto_optimizer import AutoOptimizer
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Optimization Loop")
    parser.add_argument("--target-r2", type=float, default=0.99, help="Target R2 score")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum iterations")
    parser.add_argument("--interval", type=int, default=0, help="Interval between iterations (seconds)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("AUTO OPTIMIZATION LOOP")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target R2: {args.target_r2}")
    print(f"Max Iterations: {args.max_iterations}")
    print("=" * 70)
    
    optimizer = AutoOptimizer()
    
    result = optimizer.run_loop(
        max_iterations=args.max_iterations,
        target_r2=args.target_r2
    )
    
    if result:
        print("\n" + "=" * 70)
        print("FINAL BEST RESULT")
        print("=" * 70)
        print(f"Model: {result['model']}")
        print(f"R2: {result['metrics']['r2']:.4f}")
        print(f"RMSE: {result['metrics']['rmse']:.4f}")
        print(f"MAE: {result['metrics']['mae']:.4f}")
    
    # Show history
    print("\n" + "=" * 70)
    print("OPTIMIZATION HISTORY")
    print("=" * 70)
    
    history = optimizer.history.history
    if history:
        print(f"{'#':<4} {'Model':<20} {'R2':>10} {'RMSE':>10} {'Time':<20}")
        print("-" * 70)
        for i, entry in enumerate(history[-20:], 1):  # Last 20
            metrics = entry.get('metrics', {})
            print(f"{i:<4} {entry['model']:<20} {metrics.get('r2', 0):>10.4f} {metrics.get('rmse', 0):>10.4f} {entry.get('timestamp', '')[:19]}")


if __name__ == "__main__":
    main()
