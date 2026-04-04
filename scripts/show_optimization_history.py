"""
Show Optimization History
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    history_path = Path("logs/optimization_history.json")
    
    if not history_path.exists():
        print("No optimization history found.")
        return
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    if not history:
        print("Optimization history is empty.")
        return
    
    print("=" * 80)
    print("OPTIMIZATION HISTORY")
    print("=" * 80)
    print(f"Total entries: {len(history)}")
    
    # Table header
    print(f"\n{'#':<4} {'Model':<20} {'R2':>10} {'RMSE':>10} {'MAE':>10} {'Time(s)':>10} {'Timestamp':<20}")
    print("-" * 80)
    
    for i, entry in enumerate(history, 1):
        metrics = entry.get('metrics', {})
        print(f"{i:<4} {entry['model']:<20} {metrics.get('r2', 0):>10.4f} {metrics.get('rmse', 0):>10.4f} {metrics.get('mae', 0):>10.4f} {metrics.get('train_time', 0):>10.1f} {entry.get('timestamp', '')[:19]}")
    
    # Best result
    best = max(history, key=lambda x: x.get('metrics', {}).get('r2', 0))
    
    print("\n" + "=" * 80)
    print("BEST RESULT")
    print("=" * 80)
    print(f"Model: {best['model']}")
    print(f"R2: {best['metrics']['r2']:.4f}")
    print(f"RMSE: {best['metrics']['rmse']:.4f}")
    print(f"MAE: {best['metrics']['mae']:.4f}")
    print(f"Params: {best.get('params', {})}")
    
    # Progress chart
    print("\n" + "=" * 80)
    print("R2 PROGRESS")
    print("=" * 80)
    
    r2_values = [e['metrics']['r2'] for e in history]
    min_r2 = min(r2_values)
    max_r2 = max(r2_values)
    
    if max_r2 > min_r2:
        for i, r2 in enumerate(r2_values[-20:]):  # Last 20
            bar_len = int((r2 - min_r2) / (max_r2 - min_r2) * 40)
            bar = '#' * bar_len
            print(f"{i+1:3d} | {bar:<40} {r2:.4f}")


if __name__ == "__main__":
    main()
