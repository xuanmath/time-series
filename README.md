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

| 模型 | 说明 | 适用场景 |
|------|------|----------|
| **ARIMA/SARIMA** | 经典时间序列模型 | 单变量、短期预测 |
| **Prophet** | Facebook 开源模型 | 季节性数据、趋势预测 |
| **LSTM** | 长短期记忆网络 | 多变量、长期依赖 |
| **Transformer** | 注意力机制模型 | 多变量、复杂模式 |

## 风电功率预测

本项目包含完整的风电功率预测流程：

### 数据特征
- 时间范围: 2017-2021 (5年小时数据)
- 特征: 温度、湿度、风速、风向等
- 目标: 发电功率预测

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行风电预测 (Transformer)
python scripts/train_wind_power.py --model transformer

# 运行风电预测 (LSTM)
python scripts/train_wind_power.py --model lstm --seq-len 48
```

### 可调参数

```bash
# 修改预测窗口长度
python scripts/train_wind_power.py --seq-len 48  # 使用过去48小时预测

# 修改配置文件
# 编辑 configs/wind_power.yaml
```

## 模型配置

编辑 `configs/wind_power.yaml`:

```yaml
model:
  name: "transformer"  # 或 "lstm"
  
  transformer:
    seq_len: 24          # 输入序列长度
    d_model: 64          # 模型维度
    nhead: 4             # 注意力头数
    num_encoder_layers: 2
    epochs: 100
    batch_size: 64
```

## 项目特点

- ✅ 多变量时间序列支持
- ✅ 自动特征工程（时间特征、风力特征）
- ✅ 数据标准化与预处理
- ✅ 早停与模型保存
- ✅ 多种评估指标

## License

MIT
