# Lab 数据与 Ax 数据可视化比较工具

这套工具用于可视化比较来自 Vicon 系统（Lab 数据）和 Axivity 传感器（Ax 数据）的加速度数据。

## 文件说明

1. **`visualize_lab_ax_comparison.py`** - 主要的可视化脚本
2. **`file_browser.py`** - 交互式文件浏览器，帮助选择数据文件
3. **`README_visualization.md`** - 本说明文件

## 功能特性

- 📊 **时域比较**: 同时显示 X、Y、Z 轴和合成加速度的时间序列
- 🔍 **频域分析**: 对比两种数据源的频谱特性
- 📈 **统计信息**: 显示采样率、数据范围等统计信息
- 🎨 **美观图表**: 使用 seaborn 样式的现代化可视化界面
- 💾 **自动保存**: 自动保存高分辨率的可视化图片

## 使用方法

### 方法一：交互式文件浏览器（推荐）

```bash
python file_browser.py
```

这个脚本会：
1. 自动扫描配置的数据目录
2. 列出所有可用的 Lab 和 Ax 数据文件
3. 提供交互式选择界面
4. 预览文件内容
5. 自动生成可视化

### 方法二：直接指定文件路径

```bash
python visualize_lab_ax_comparison.py --lab_file <lab_file_path> --ax_file <ax_file_path>
```

示例:
```bash
python visualize_lab_ax_comparison.py \
  --lab_file "/path/to/lab_data.joblib" \
  --ax_file "/path/to/ax_data.csv" \
  --save_dir "./output"
```

### 方法三：交互式输入

```bash
python visualize_lab_ax_comparison.py
```

然后按提示输入文件路径。

## 数据格式要求

### Lab 数据 (.joblib 文件)
- 支持包含 'left' 和 'right' 键的字典格式
- 每个部分应包含以下字段：
  - `accelX`: X轴加速度数据
  - `accelY`: Y轴加速度数据  
  - `accelZ`: Z轴加速度数据
  - `accel`: 合成加速度（可选，会自动计算）

### Ax 数据 (.csv 文件)
- 第一列：时间戳（Unix时间戳或时间字符串）
- 第二到四列：X、Y、Z轴加速度数据
- 标准格式：`Timestamp,X,Y,Z`

## 输出文件

可视化结果会保存在指定目录（默认为 `./visualizations/`）：

1. **`{lab_name}_vs_{ax_name}_comparison.png`** - 时域比较图
2. **`{lab_name}_vs_{ax_name}_frequency.png`** - 频域分析图

## 可视化内容

### 时域比较图 (2x2 子图)
- **左上**: X轴加速度对比
- **右上**: Y轴加速度对比  
- **左下**: Z轴加速度对比
- **右下**: 合成加速度对比

每个子图都会显示：
- Lab 数据（1500Hz，细线，半透明）
- Ax 数据（20Hz，粗线，不透明）
- 网格和图例

### 频域分析图 (1x2 子图)
- **左**: Lab 数据频谱
- **右**: Ax 数据频谱
- 使用对数刻度显示幅度

### 控制台统计信息
- 数据持续时间
- 总样本数
- 实际 vs 期望采样率
- 各轴数据范围

## 配置参数

在 `config.py` 中可以调整以下参数：

```python
LAB_SAMPLING_RATE = 1500  # Lab 数据采样率 (Hz)
AW_SAMPLING_RATE = 20     # Ax 数据采样率 (Hz)
WINDOW_SIZE = 5           # 窗口大小 (秒)
lab_data_dir = "..."      # Lab 数据目录
ax_data_dir = "..."       # Ax 数据目录
```

## 依赖库

```bash
pip install pandas numpy matplotlib seaborn joblib loguru tqdm
```

## 故障排除

### 常见问题

1. **找不到数据文件**
   - 检查 `config.py` 中的路径设置
   - 确保文件权限正确

2. **内存不足**
   - 脚本会自动限制显示时间到30秒
   - 对于超大文件，考虑先截取部分数据

3. **时间戳格式错误**
   - Ax 数据支持多种时间戳格式
   - 如有问题，可以手动转换为 Unix 时间戳

4. **图表显示问题**
   - 确保已安装 matplotlib 后端
   - 在服务器环境可能需要使用 `Agg` 后端

### 调试技巧

- 使用 `file_browser.py` 的预览功能查看文件结构
- 检查控制台输出的统计信息
- 确认采样率设置正确

## 扩展功能

脚本设计为模块化，可以轻松扩展：

- 添加新的数据源支持
- 实现更多分析功能（如相关性分析）
- 自定义可视化样式
- 批量处理多个文件

## 示例输出

```
=== Data Statistics ===
Lab Data:
  Duration: 15.32 seconds
  Total samples: 22980
  Expected sampling rate: 1500 Hz
  Actual sampling rate: 1500.00 Hz
  X-axis range: [-2.456, 3.123]
  Y-axis range: [-1.789, 2.567]
  Z-axis range: [-0.987, 4.321]

Ax Data:
  Duration: 15.35 seconds
  Total samples: 307
  Expected sampling rate: 20 Hz
  Actual sampling rate: 20.00 Hz
  X-axis range: [-2.234, 2.987]
  Y-axis range: [-1.654, 2.345]
  Z-axis range: [-1.123, 4.123]
```
