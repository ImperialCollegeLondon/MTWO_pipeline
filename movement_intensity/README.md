# Movement Intensity Analysis

本模块提供了基于实验室加速度数据的运动强度分析功能。

## 主要功能

### 1. 数据统计分析
- `analyze_acceleration_statistics()`: 分析目录下所有lab数据文件的加速度统计信息
- 计算全局统计量（均值、标准差、百分位数等）

### 2. 强度阈值计算
- `calculate_intensity_thresholds()`: 基于统计数据计算运动强度阈值
- 支持三种方法：
  - `'percentile'`: 基于百分位数（推荐）
  - `'std'`: 基于标准差
  - `'hybrid'`: 混合方法

### 3. 运动强度分类
- `classify_movement_intensity()`: 将加速度数据分类为5个强度等级
- 强度等级：`very_low`, `low`, `medium`, `high`, `very_high`

### 4. 单文件分析
- `analyze_movement_intensity()`: 分析单个lab数据文件的运动强度

### 5. 批量分析
- `batch_analyze_movement_intensity()`: 批量分析目录下所有文件

## 使用方法

### 快速开始

```python
from get_intensity import batch_analyze_movement_intensity

# 批量分析所有文件
results = batch_analyze_movement_intensity(
    lab_data_dir="path/to/your/lab/data",
    output_file="intensity_results.csv",
    side='right',
    aggregation='mean',
    threshold_method='percentile'
)

print(results['intensity_level'].value_counts())
```

### 详细使用步骤

1. **分析数据统计**：
```python
from get_intensity import analyze_acceleration_statistics, calculate_intensity_thresholds

# 分析所有文件的统计信息
stats = analyze_acceleration_statistics("path/to/lab/data", side='right')

# 计算强度阈值
thresholds = calculate_intensity_thresholds(stats, method='percentile')
```

2. **分析单个文件**：
```python
from get_intensity import analyze_movement_intensity

result = analyze_movement_intensity(
    lab_file_path="path/to/single/file.joblib",
    thresholds=thresholds,
    side='right',
    aggregation='mean'
)

print(f"Intensity: {result['intensity_level']}")
print(f"Confidence: {result['confidence_score']:.3f}")
```

### 参数说明

- **side**: `'left'` 或 `'right'`，选择分析哪条腿的数据
- **aggregation**: 加速度聚合方法
  - `'mean'`: 平均值（推荐）
  - `'median'`: 中位数
  - `'percentile_90'`: 90百分位数
  - `'std'`: 标准差
- **threshold_method**: 阈值计算方法
  - `'percentile'`: 基于百分位数（推荐）
  - `'std'`: 基于标准差
  - `'hybrid'`: 混合方法

## 输出结果

分析结果包含以下信息：
- `intensity_level`: 运动强度等级
- `confidence_score`: 置信度分数（0-1）
- `aggregated_value`: 聚合后的加速度值
- `duration_seconds`: 数据持续时间
- `mean_acceleration`: 平均加速度
- `std_acceleration`: 加速度标准差

## 示例脚本

运行 `example_usage.py` 查看完整的使用示例：

```bash
python example_usage.py
```

或直接运行主脚本：

```bash
python get_intensity.py
```

## 注意事项

1. 确保lab数据文件格式正确（.joblib格式）
2. 数据文件应包含 `'accelX'`, `'accelY'`, `'accelZ'` 字段
3. 建议先运行批量分析了解数据分布，再进行单文件分析
4. 置信度分数越高表示分类结果越可靠
5. 默认采样率为1500Hz，如有不同请修改 `LAB_SAMPLING_RATE` 常量
