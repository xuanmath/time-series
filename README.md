# Time Series Analysis

时间序列分析项目 - 包含多种预测模型和特征工程方法

## 项目结构

```
time-series/
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── results/           # 预测结果
├── src/                    # 源代码
│   ├── models/            # 模型实现
│   ├── features/          # 特征工程
│   └── utils/             # 工具函数
├── notebooks/              # Jupyter notebooks
├── tests/                  # 单元测试
├── scripts/                # 脚本文件
└── configs/                # 配置文件
```

## 支持的模型

- ARIMA / SARIMA
- Prophet
- LSTM / GRU
- Transformer
- XGBoost / LightGBM

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行示例
python scripts/train.py --config configs/config.yaml
```

## License

MIT
