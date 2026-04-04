# AutoML Optimization System

## 目标
自动化模型优化闭环：
```
指标定义 → 自动优化 → 自动验证 → 自动提交/回滚 → 循环迭代
```

## 核心组件

### 1. 指标定义 (`configs/metrics.yaml`)
- 主指标: R² (越大越好)
- 辅助指标: RMSE, MAE, SMAPE
- 目标阈值: R² > 0.98

### 2. 自动优化 (`scripts/auto_optimizer.py`)
- 超参数搜索: Grid Search / Random Search / Bayesian
- 模型选择: 自动对比多个模型
- 特征工程: 自动添加/删除特征

### 3. 自动验证 (`scripts/validate.py`)
- 交叉验证
- 时序数据切分验证
- 指标计算与记录

### 4. Git自动化 (`scripts/git_manager.py`)
- 提交改进版本
- 回滚失败的优化
- 记录优化历史

### 5. 循环迭代 (`scripts/run_optimization_loop.py`)
- 持续运行优化
- 达到目标后停止
- 定期汇报进度

## 使用方法

```bash
# 单次优化
python scripts/run_optimization.py

# 持续优化循环
python scripts/run_optimization_loop.py --target-r2 0.99 --max-iterations 50

# 查看优化历史
python scripts/show_optimization_history.py
```

## 目录结构
```
time-series/
├── configs/
│   └── metrics.yaml          # 指标配置
├── scripts/
│   ├── auto_optimizer.py     # 核心优化器
│   ├── git_manager.py        # Git自动化
│   ├── run_optimization.py   # 单次优化
│   └── run_optimization_loop.py  # 循环优化
├── logs/
│   └── optimization_history.json  # 优化历史
└── results/
    └── best_model/           # 最佳模型
```
