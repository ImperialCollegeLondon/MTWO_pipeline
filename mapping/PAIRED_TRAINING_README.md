# 配对训练数据映射模型 - 修改说明

## 概述

我已经成功修改了 `mapping.py` 文件，使其能够加载和处理由 `build_training_set.py` 生成的配对训练数据。新的训练流程能够自动处理多个配对样本，应用不同的坐标系对齐方法，并训练映射模型。

## 主要修改内容

### 1. 新增导入模块
```python
import pickle  # 用于加载配对训练数据
```

### 2. 新增核心函数

#### `load_paired_training_data(pickle_path: str) -> list`
- 从pickle文件加载配对训练数据
- 返回训练样本列表

#### `convert_sample_to_dataframes(sample: dict) -> tuple`
- 将单个配对样本转换为AW和Vicon的DataFrame格式
- 便于后续处理和对齐操作

#### `process_paired_samples_with_alignment(training_samples: list, alignment_method: str) -> tuple`
- 批量处理所有配对样本
- 支持三种对齐方法：`rotation_matrix`、`procrustes`、`none`
- 返回对齐后的数据和对齐信息

#### `calculate_rotation_matrix_from_sample(vicon_df: pd.DataFrame, aw_df: pd.DataFrame) -> np.ndarray`
- 基于单个样本计算旋转矩阵
- 使用平均加速度向量（重力方向）进行对齐

#### `train_mapping_model_from_paired_data(training_samples: list, alignment_method: str) -> tuple`
- 使用配对数据训练映射模型的主函数
- 集成数据处理、对齐和模型训练的完整流程
- 返回训练好的模型、对齐信息和训练指标

### 3. 新的主程序流程

完全重写了 `__main__` 部分，新流程包括：

1. **加载配对训练数据**
   - 从 `paired_training_data/paired_training_data.pkl` 加载数据
   - 显示样本概览和统计信息

2. **多方法训练和比较**
   - 使用三种对齐方法分别训练模型
   - 自动保存模型文件和相关信息到不同目录

3. **性能比较和可视化**
   - 比较不同对齐方法的性能
   - 生成性能对比图表
   - 自动选择最佳方法

4. **模型测试**
   - 使用最佳模型进行预测测试
   - 生成预测结果可视化

## 文件结构

训练完成后会生成以下文件结构：

```
mapping/
├── mapping_models_rotation_matrix/
│   ├── mapping_model.joblib        # 旋转矩阵方法训练的模型
│   ├── alignment_info.pkl         # 对齐信息
│   └── training_metrics.pkl       # 训练指标
├── mapping_models_procrustes/
│   ├── mapping_model.joblib        # Procrustes方法训练的模型
│   ├── alignment_info.pkl
│   └── training_metrics.pkl
├── mapping_models_none/
│   ├── mapping_model.joblib        # 无对齐方法训练的模型
│   ├── alignment_info.pkl
│   └── training_metrics.pkl
├── model_performance_comparison.png # 性能对比图
└── prediction_test_[best_method].png # 最佳模型测试结果图
```

## 使用方法

### 1. 生成配对训练数据
```bash
python build_training_set.py
```

### 2. 训练映射模型
```bash
python mapping.py
```

### 3. 查看使用示例
```bash
python paired_training_demo.py
```

### 4. 使用训练好的模型
```python
import joblib
import numpy as np

# 加载最佳模型
model = joblib.load('mapping_models_rotation_matrix/mapping_model.joblib')

# 准备Vicon输入数据 (N x 3)
vicon_input = your_vicon_data[['accelX', 'accelY', 'accelZ']].values

# 预测Apple Watch数据
aw_predicted = model.predict(vicon_input)
```

## 优势和改进

### 相比原方法的优势：

1. **自动化处理**：无需手动处理每个数据文件
2. **批量训练**：一次性处理所有配对样本
3. **多方法比较**：自动比较不同对齐方法的效果
4. **完整评估**：提供详细的训练指标和可视化
5. **标准化流程**：统一的数据格式和处理流程

### 技术改进：

1. **数据同步**：确保AW和Vicon数据完全对齐
2. **坐标系对齐**：支持多种对齐方法
3. **性能评估**：comprehensive的模型评估指标
4. **可视化**：直观的性能比较和预测结果展示
5. **模块化设计**：易于扩展和维护

## 配置参数

主要参数都已经内置在代码中，但可以通过修改以下变量来调整：

- `paired_data_path`: 配对训练数据的路径
- `alignment_methods`: 要使用的对齐方法列表
- 模型保存目录前缀
- 可视化设置

## 注意事项

1. 确保已经运行 `build_training_set.py` 生成配对训练数据
2. 配对数据文件路径必须正确
3. 足够的磁盘空间用于保存模型和图表
4. 如果数据量很大，训练可能需要一些时间

## 后续扩展

这个框架很容易扩展，可以：

1. 添加更多对齐方法
2. 使用不同的机器学习模型（如随机森林、神经网络）
3. 添加交叉验证
4. 实现在线学习功能
5. 添加更多评估指标

修改后的代码保持了原有功能的同时，大大提升了自动化程度和易用性。
