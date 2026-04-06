"""
Test Sliding Window Preprocessing - 测试滑动窗口预处理

测试多尺度窗口和步长调整功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.compare_pytorch import prepare_data

def test_single_window():
    """Test single window with different strides."""
    print("\n" + "="*60)
    print("TEST 1: Single Window with Different Strides")
    print("="*60)
    
    data_path = project_root / "dataset/Location1.csv"
    
    # Stride = 1 (every step)
    print("\n--- Stride = 1 (dense sequences) ---")
    data1 = prepare_data(str(data_path), seq_len=24, stride=1, log_preprocess=True)
    print(f"Result keys: {list(data1.keys())}")
    
    # Stride = 6 (every 6 hours)
    print("\n--- Stride = 6 (6-hour intervals) ---")
    data6 = prepare_data(str(data_path), seq_len=24, stride=6, log_preprocess=True)
    
    # Stride = 24 (daily samples)
    print("\n--- Stride = 24 (daily samples) ---")
    data24 = prepare_data(str(data_path), seq_len=24, stride=24, log_preprocess=True)
    
    print("\n[Comparison]")
    print(f"{'Stride':<10}{'Train Seqs':<15}{'Test Seqs':<15}{'Compression':<15}")
    print("-"*55)
    print(f"1         {len(data1['X_train_seq']):<15}{len(data1['X_test_seq']):<15}{'1x (full)':<15}")
    print(f"6         {len(data6['X_train_seq']):<15}{len(data6['X_test_seq']):<15}{'6x smaller':<15}")
    print(f"24        {len(data24['X_train_seq']):<15}{len(data24['X_test_seq']):<15}{'24x smaller':<15}")


def test_multi_scale():
    """Test multi-scale windows."""
    print("\n" + "="*60)
    print("TEST 2: Multi-Scale Windows")
    print("="*60)
    
    data_path = project_root / "dataset/Location1.csv"
    
    # Multi-scale: 6h, 12h, 24h, 48h windows
    seq_lens = [6, 12, 24, 48]
    print(f"\nMulti-scale windows: {seq_lens} hours")
    
    data = prepare_data(
        str(data_path), 
        seq_len=24, 
        stride=1, 
        multi_scale=True, 
        seq_lens=seq_lens,
        log_preprocess=True
    )
    
    print("\n[Multi-scale data]")
    ms = data['multi_scale']
    print(f"Available scales: {list(ms['train'].keys())}")
    
    for key in ms['train']:
        if key.startswith('scale_'):
            sl = ms['train'][key]['seq_len']
            n_train = len(ms['train'][key]['X'])
            n_test = len(ms['test'][key]['X'])
            shape = ms['train'][key]['X'].shape
            print(f"  {key}: train={n_train}, test={n_test}, shape={shape}")
    
    # Multi-scale aggregated features
    print("\n[Multi-scale aggregated features]")
    ms_train = ms['train']['multi_scale']
    ms_test = ms['test']['multi_scale']
    print(f"  X shape: {ms_train['X'].shape}")
    print(f"  Features per scale: {ms_train['n_features_per_scale']}")
    print(f"  Total features: {ms_train['X'].shape[1]} (= {len(seq_lens)} scales x {ms_train['n_features_per_scale']} features)")


def test_different_windows():
    """Test different window sizes."""
    print("\n" + "="*60)
    print("TEST 3: Different Window Sizes")
    print("="*60)
    
    data_path = project_root / "dataset/Location1.csv"
    
    windows = [6, 12, 24, 48, 72, 168]  # 6h, 12h, 1d, 2d, 3d, 1w
    
    print(f"\n{'Window':<12}{'Hours':<10}{'Train Seqs':<15}{'Test Seqs':<15}{'Seq Shape':<20}")
    print("-"*72)
    
    for w in windows:
        data = prepare_data(str(data_path), seq_len=w, stride=1, log_preprocess=False)
        hours = w
        train_seqs = len(data['X_train_seq'])
        test_seqs = len(data['X_test_seq'])
        shape = str(data['X_train_seq'].shape)
        print(f"{w:<12}{hours:<10}{train_seqs:<15}{test_seqs:<15}{shape:<20}")


def main():
    print("\n" + "#"*60)
    print("# SLIDING WINDOW PREPROCESSING TEST")
    print("#"*60)
    
    test_single_window()
    test_multi_scale()
    test_different_windows()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()