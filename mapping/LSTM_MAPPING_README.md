# LSTM映射模型文档 (LSTM Mapping Model)

## 概述 (Overview)

`lstm_mapping.py` 是一个基于长短期记忆网络（LSTM）的深度学习模型，用于将Vicon运动捕捉系统的加速度数据映射到Apple Watch的加速度数据。该模型通过学习两个设备之间的复杂时序关系，实现比传统线性回归更精确的数据转换。

## 主要功能 (Main Features)

- **时序建模**: 利用LSTM网络学习加速度数据的时序依赖关系
- **多种对齐方法**: 支持旋转矩阵、Procrustes分析和无对齐三种坐标系统对齐方法
- **自动化训练**: 端到端的模型训练、验证和测试流程
- **性能评估**: 全面的模型性能评估和可视化
- **模型持久化**: 完整的模型保存和加载机制

## 文件结构 (File Structure)

### 核心类 (Core Classes)

#### `LSTMMappingModel`
LSTM映射模型的主要类，包含完整的深度学习模型生命周期：

**初始化参数:**
- `sequence_length`: LSTM输入序列长度（默认：50）
- `lstm_units`: 每层LSTM单元数量（默认：64）
- `dropout_rate`: Dropout正则化率（默认：0.2）

**主要方法:**
- `create_sequences()`: 创建LSTM训练序列
- `prepare_data()`: 数据预处理和标准化
- `build_model()`: 构建LSTM网络架构
- `train()`: 模型训练
- `predict()`: 模型预测
- `evaluate()`: 模型评估
- `save_model()`: 保存训练好的模型
- `load_model()`: 加载预训练模型

### 核心函数 (Core Functions)

#### `process_paired_samples_with_alignment_lstm()`
处理配对样本并应用坐标系统对齐，专门为LSTM训练优化。

**参数:**
- `training_samples`: 配对训练样本列表
- `alignment_method`: 对齐方法（'rotation_matrix', 'procrustes', 'none'）

**返回:**
- 对齐后的Apple Watch数据列表
- 对齐后的Vicon数据列表
- 对齐信息字典

#### `train_lstm_mapping_model()`
完整的LSTM模型训练管道。

**参数:**
- `training_samples`: 训练样本
- `alignment_method`: 坐标对齐方法
- `sequence_length`: 序列长度
- `lstm_units`: LSTM单元数
- `dropout_rate`: Dropout率
- `epochs`: 训练轮数
- `batch_size`: 批次大小

#### `visualize_lstm_training_history()`
可视化LSTM训练历史，包括损失函数和评估指标的变化。

#### `test_lstm_model_on_sample()`
在单个样本上测试训练好的LSTM模型。

## LSTM网络架构 (LSTM Architecture)

```
输入层: (sequence_length, 3) - 3轴加速度序列
    ↓
LSTM层1: 64单元, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM层2: 64单元, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM层3: 64单元, return_sequences=False
    ↓
Dropout: 0.2
    ↓
全连接层: 32单元, ReLU激活
    ↓
Dropout: 0.2
    ↓
输出层: 3单元, 线性激活 (X, Y, Z加速度)
```

## 工作流程 (Workflow)

### 1. 数据准备阶段 (Data Preparation)

```python
# 加载配对训练数据
training_samples = load_paired_training_data("paired_training_data/paired_training_data.pkl")

# 处理样本并应用坐标对齐
aligned_aw_data, aligned_vicon_data, alignment_info = process_paired_samples_with_alignment_lstm(
    training_samples, alignment_method='rotation_matrix'
)
```

### 2. 模型训练阶段 (Model Training)

```python
# 创建LSTM模型
lstm_model = LSTMMappingModel(sequence_length=50, lstm_units=64, dropout_rate=0.2)

# 准备训练数据
X_sequences, y_sequences = lstm_model.prepare_data(aligned_vicon_data, aligned_aw_data)

# 训练模型
training_history = lstm_model.train(X_sequences, y_sequences, epochs=100, batch_size=32)
```

### 3. 模型评估阶段 (Model Evaluation)

```python
# 评估模型性能
metrics = lstm_model.evaluate(X_sequences, y_sequences)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

### 4. 模型保存阶段 (Model Saving)

```python
# 保存完整模型
lstm_model.save_model("lstm_mapping_models_rotation_matrix")
```

## 数据流程 (Data Flow)

### 输入数据处理
1. **原始数据**: Vicon和Apple Watch的配对加速度数据
2. **坐标对齐**: 应用选定的对齐方法统一坐标系
3. **数据标准化**: 使用MinMaxScaler将数据缩放到[-1, 1]范围
4. **序列创建**: 将连续数据切分为固定长度的时间序列
5. **序列对齐**: 调整输入和目标序列的时间对齐

### LSTM训练数据格式
- **输入 (X)**: `(n_sequences, sequence_length, 3)` - Vicon加速度序列
- **目标 (y)**: `(n_sequences, 3)` - Apple Watch加速度值

## 对齐方法比较 (Alignment Methods)

### 1. 旋转矩阵法 (Rotation Matrix)
使用预定义的旋转矩阵对齐坐标系统：
```python
rotation_matrix = np.array([[ 0.58377147,  0.28237329,  0.76123334],
                           [ 0.36897715,  0.74289845, -0.55853179],
                           [-0.72323352,  0.60693263,  0.32949362]])
```

### 2. Procrustes分析 (Procrustes Analysis)
通过最小化两个数据集之间的距离来自动计算最优旋转矩阵。

### 3. 无对齐 (None)
直接使用原始坐标系统，不进行任何变换。

## 模型性能指标 (Performance Metrics)

### 整体指标
- **RMSE (均方根误差)**: 预测值与真实值之间的均方根误差
- **R² Score (决定系数)**: 模型解释数据变异性的比例
- **MAE (平均绝对误差)**: 预测值与真实值的平均绝对差值

### 分轴指标
- **X_rmse, Y_rmse, Z_rmse**: 各轴的RMSE
- **X_r2, Y_r2, Z_r2**: 各轴的R²分数

## 输出文件 (Output Files)

### 模型文件
- `lstm_model.h5`: 训练好的Keras模型
- `scaler_input.joblib`: 输入数据标准化器
- `scaler_output.joblib`: 输出数据标准化器
- `model_params.pkl`: 模型超参数
- `training_history.pkl`: 训练历史记录
- `alignment_info.pkl`: 坐标对齐信息
- `training_metrics.pkl`: 训练评估指标

### 可视化文件
- `training_history.png`: 训练历史可视化
- `lstm_model_performance_comparison.png`: 不同对齐方法性能比较
- `lstm_prediction_test_{method}.png`: 预测结果可视化

## 使用示例 (Usage Example)

### 完整训练流程
```python
# 运行完整的LSTM训练流程
python lstm_mapping.py
```

### 自定义训练
```python
from lstm_mapping import LSTMMappingModel, train_lstm_mapping_model

# 自定义参数训练
lstm_model, alignment_info, metrics = train_lstm_mapping_model(
    training_samples,
    alignment_method='procrustes',
    sequence_length=30,
    lstm_units=128,
    epochs=150
)
```

### 加载预训练模型
```python
from lstm_mapping import LSTMMappingModel

# 加载模型
lstm_model = LSTMMappingModel()
lstm_model.load_model("lstm_mapping_models_rotation_matrix")

# 进行预测
predictions = lstm_model.predict(X_sequences)
```

## 依赖项 (Dependencies)

### 核心依赖
- `tensorflow>=2.0`: 深度学习框架
- `numpy>=1.19.0`: 数值计算
- `pandas>=1.3.0`: 数据处理
- `scikit-learn>=0.24.0`: 机器学习工具
- `matplotlib>=3.3.0`: 数据可视化

### 项目特定依赖
- `mapping.py`: 基础映射功能模块
- `config.py`: 配置和日志模块

## 性能优化建议 (Performance Optimization)

### 超参数调优
1. **序列长度**: 根据数据特性调整，较长序列可能捕获更多时序信息
2. **LSTM单元数**: 增加可提高模型容量，但可能导致过拟合
3. **Dropout率**: 根据验证性能调整正则化强度
4. **学习率**: 使用学习率调度器优化训练过程

### 训练策略
1. **早停机制**: 防止过拟合，在验证损失不再改善时停止训练
2. **学习率衰减**: 动态调整学习率以提高收敛稳定性
3. **批次大小**: 根据GPU内存和数据大小选择合适的批次大小

## 故障排除 (Troubleshooting)

### 常见问题

1. **内存不足**: 减少序列长度或批次大小
2. **训练不收敛**: 调整学习率或网络架构
3. **过拟合**: 增加Dropout率或使用更多正则化
4. **数据不平衡**: 检查不同样本的数据分布

### 调试技巧
1. 检查数据形状和范围
2. 监控训练和验证损失曲线
3. 可视化预测结果
4. 分析各轴的性能差异

## 扩展功能 (Extensions)

### 可能的改进方向
1. **注意力机制**: 添加注意力层提高模型性能
2. **多尺度特征**: 使用不同时间窗口的特征
3. **集成学习**: 结合多个模型的预测结果
4. **实时预测**: 优化模型以支持实时数据流处理

---

**作者**: MTWO Pipeline Team  
**最后更新**: 2025年7月31日  
**版本**: 1.0
