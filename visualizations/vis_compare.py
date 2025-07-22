'''
可视化脚本：比较 Lab 数据（Vicon）和 Aw 数据（Axivity）
用户可以指定具体的数据文件进行可视化比较

使用方法:
python visualize_lab_ax_comparison.py --lab_file <lab_file_path> --ax_file <ax_file_path>
或者直接运行脚本，然后按提示输入文件路径
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import argparse
import datetime
from pathlib import Path
from loguru import logger
from dataTransformer.filter import filter

# Configure loguru logger
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

# Add the parent directory to the Python path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LAB_SAMPLING_RATE, AW_SAMPLING_RATE

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_lab_file(lab_file_path, side='right'):
    logger.info(f"Loading lab data from: {lab_file_path}")
    
    if not os.path.exists(lab_file_path):
        raise FileNotFoundError(f"Lab file not found: {lab_file_path}")

    lab_data = joblib.load(lab_file_path)
    if 'left' in lab_data and 'right' in lab_data:
        data = lab_data[side]
        logger.warning(f"Using {side} leg data from lab file")
    else:
        logger.warning("Lab file does not contain left/right data, using the first one in the structure.")
        data = list(lab_data.values())[0] # 选第一个
    
    # Extract acceleration data
    accel_x = data.get('accelX', [])
    accel_y = data.get('accelY', [])
    accel_z = data.get('accelZ', [])

    assert len(accel_x) == len(accel_y) == len(accel_z), "Acceleration data arrays must have the same length."
    assert len(accel_x) > 0, "Acceleration data cannot be empty."
    
    logger.info("Applying Butterworth filter to lab data...")
    accel_x_filtered = filter(np.array(accel_x), sampling_rate=LAB_SAMPLING_RATE)
    accel_y_filtered = filter(np.array(accel_y), sampling_rate=LAB_SAMPLING_RATE)
    accel_z_filtered = filter(np.array(accel_z), sampling_rate=LAB_SAMPLING_RATE)

    accel_data = {
        'timestamp': np.arange(len(accel_x_filtered)) / LAB_SAMPLING_RATE,
        'accelX': accel_x_filtered,
        'accelY': accel_y_filtered,
        'accelZ': accel_z_filtered,
        'accel': data.get('accel', [])
    }
    
    # Compute synthetic acceleration using filtered components
    accel_data['accel'] = np.sqrt(accel_x_filtered**2 + accel_y_filtered**2 + accel_z_filtered**2)
        
    return accel_data

def load_ax_file(ax_file_path):
    """
    加载 Aw 数据文件 (.csv)
    返回处理后的数据用于可视化
    """
    logger.info(f"Loading Aw Data from: {ax_file_path}")
    
    if not os.path.exists(ax_file_path):
        raise FileNotFoundError(f"Aw file not found: {ax_file_path}")
    
    # 读取 CSV 文件
    ax_data = pd.read_csv(ax_file_path)
    
    # 标准化列名
    if ax_data.columns[0].lower() in ['timestamp', 'time']:
        ax_data.columns = ["Timestamp", "X", "Y", "Z"]
    else:
        # 假设前四列是 timestamp, X, Y, Z
        ax_data.columns = ["Timestamp", "X", "Y", "Z"]
    
    # 处理时间戳
    if ax_data["Timestamp"].dtype == 'object':
        # 如果是字符串格式的时间戳
        try:
            ax_data["Timestamp"] = pd.to_datetime(ax_data["Timestamp"])
        except:
            # 如果无法解析，假设是 Unix 时间戳
            ax_data["Timestamp"] = ax_data["Timestamp"].map(
                lambda t: datetime.datetime.fromtimestamp(t) if isinstance(t, (int, float)) else t
            )
    elif ax_data["Timestamp"].dtype in ['int64', 'float64']:
        # Unix 时间戳
        ax_data["Timestamp"] = ax_data["Timestamp"].map(
            lambda t: datetime.datetime.fromtimestamp(t)
        )
    
    # 创建相对时间戳（从 0 开始的秒数）
    start_time = ax_data["Timestamp"].iloc[0]
    ax_data["RelativeTime"] = (ax_data["Timestamp"] - start_time).dt.total_seconds()
    
    # 应用滤波器到Ax数据
    logger.info("Applying Butterworth filter to Aw Data...")
    accel_x_filtered = filter(ax_data["X"].values)
    accel_y_filtered = filter(ax_data["Y"].values)
    accel_z_filtered = filter(ax_data["Z"].values)
    
    # 计算滤波后的合成加速度
    accel_magnitude_filtered = np.sqrt(accel_x_filtered**2 + accel_y_filtered**2 + accel_z_filtered**2)
    
    accel_data = {
        'timestamp': ax_data["RelativeTime"].values,
        'accelX': accel_x_filtered,
        'accelY': accel_y_filtered,
        'accelZ': accel_z_filtered,
        'accel': accel_magnitude_filtered
    }
    
    return accel_data

def crop_aw_data(accel_data, start_time=None, end_time=None):
    """
    对Apple Watch数据进行时间裁剪
    
    Args:
        accel_data: 包含timestamp和加速度数据的字典
        start_time: 开始时间（秒），如果为None则从开头开始
        end_time: 结束时间（秒），如果为None则到结尾
    
    Returns:
        裁剪后的数据字典
    """
    logger.info(f"Cropping Apple Watch data from {start_time}s to {end_time}s")
    
    # 获取时间戳
    timestamps = np.array(accel_data['timestamp'])
    
    # 确定裁剪范围
    if start_time is None:
        start_time = timestamps[0]
    if end_time is None:
        end_time = timestamps[-1]
    if start_time is None and end_time is None:
        logger.warning("No time range specified, returning original data")
        return accel_data
    
    # 创建时间掩码
    time_mask = (timestamps >= start_time) & (timestamps <= end_time)
    
    # 检查是否有有效数据
    if not np.any(time_mask):
        logger.warning(f"No data found in time range {start_time}s to {end_time}s")
        logger.info(f"Available time range: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s")
        return accel_data
    
    # 应用裁剪
    cropped_data = {}
    for key in ['timestamp', 'accelX', 'accelY', 'accelZ', 'accel']:
        if key in accel_data:
            cropped_data[key] = np.array(accel_data[key])[time_mask]
    
    # 重新调整时间戳，从裁剪后的开始时间归零
    if 'timestamp' in cropped_data:
        cropped_data['timestamp'] = cropped_data['timestamp'] - cropped_data['timestamp'][0]
    
    logger.info(f"Cropped data: {len(cropped_data['timestamp'])} samples, duration: {cropped_data['timestamp'][-1]:.2f}s")
    
    return cropped_data

def load_aw_file(aw_file_path):
    logger.info(f"Loading Apple Watch data from: {aw_file_path}")
    
    if not os.path.exists(aw_file_path):
        raise FileNotFoundError(f"Apple Watch file not found: {aw_file_path}")
    
    aw_data = pd.read_csv(aw_file_path)
    aw_subset = aw_data[["Timestamp", "accelerationX", "accelerationY", "accelerationZ"]].copy()
    aw_subset.columns = ["Timestamp", "X", "Y", "Z"]
    aw_subset["Timestamp"] = aw_subset["Timestamp"].map(
        lambda t: datetime.datetime.fromtimestamp(t)
        )
    
    # Relative timestamp
    start_timestamp = aw_subset["Timestamp"].iloc[0]
    aw_subset["RelativeTime"] = (aw_subset["Timestamp"] - start_timestamp).dt.total_seconds()
    
    # Apply Butterworth filter
    logger.info("Applying Butterworth filter to Apple Watch data...")
    accel_x_filtered = filter(aw_subset["X"].values, order=4, cutoff_freq=5, sampling_rate=AW_SAMPLING_RATE)
    accel_y_filtered = filter(aw_subset["Y"].values, order=4, cutoff_freq=5, sampling_rate=AW_SAMPLING_RATE)
    accel_z_filtered = filter(aw_subset["Z"].values, order=4, cutoff_freq=5, sampling_rate=AW_SAMPLING_RATE)
    accel_magnitude_filtered = np.sqrt(accel_x_filtered**2 + accel_y_filtered**2 + accel_z_filtered**2)
    
    accel_data = {
        'timestamp': aw_subset["RelativeTime"].values,
        'accelX': accel_x_filtered,
        'accelY': accel_y_filtered,
        'accelZ': accel_z_filtered,
        'accel': accel_magnitude_filtered
    }

    return accel_data


def plot_comparison(data1, data2, save_path=None):
    """
    创建 Lab 数据和 Aw 数据的比较可视化
    """
    logger.info("Creating comparison visualization...")
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    fig.suptitle('Lab Data vs Aw Data Comparison', fontsize=16, fontweight='bold')
    
    # # 确定时间范围（取较短的那个）
    # max_time_lab = max(data1['timestamp'])
    # max_time_ax = max(data2['timestamp'])
    # max_time = min(max_time_lab, max_time_ax)
    
    # # 过滤数据到指定时间范围
    # lab_mask = data1['timestamp'] <= max_time
    # ax_mask = data2['timestamp'] <= max_time
    lab_mask = data1['timestamp'] >= 0 
    ax_mask = data2['timestamp'] >= 0 
    
    # 1. X轴加速度比较
    axes[0, 0].plot(data1['timestamp'][lab_mask], 
                   data1['accelX'][lab_mask], 
                   label='Lab Data', linewidth=1, alpha=0.7)
    axes[0, 0].plot(data2['timestamp'][ax_mask], 
                   data2['accelX'][ax_mask], 
                   label='Aw Data', linewidth=1)
    axes[0, 0].set_title('X-axis Acceleration')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Acceleration (g)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Y轴加速度比较
    axes[0, 1].plot(data1['timestamp'][lab_mask], 
                   data1['accelY'][lab_mask], 
                   label='Lab Data', linewidth=1, alpha=0.7)
    axes[0, 1].plot(data2['timestamp'][ax_mask], 
                   data2['accelY'][ax_mask], 
                   label='Aw Data', linewidth=1)
    axes[0, 1].set_title('Y-axis Acceleration')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Acceleration (g)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Z轴加速度比较
    axes[1, 0].plot(data1['timestamp'][lab_mask], 
                   data1['accelZ'][lab_mask], 
                   label='Lab Data', linewidth=1, alpha=0.7)
    axes[1, 0].plot(data2['timestamp'][ax_mask], 
                   data2['accelZ'][ax_mask], 
                   label='Aw Data', linewidth=1)
    axes[1, 0].set_title('Z-axis Acceleration')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Acceleration (g)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 合成加速度比较
    axes[1, 1].plot(data1['timestamp'][lab_mask], 
                   data1['accel'][lab_mask], 
                   label='Lab Data', linewidth=1, alpha=0.7)
    axes[1, 1].plot(data2['timestamp'][ax_mask], 
                   data2['accel'][ax_mask], 
                   label='Aw Data', linewidth=1)
    axes[1, 1].set_title('Magnitude Acceleration')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Acceleration (g)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")
    
    plt.show()

def special_filter_mask(freqs, magnitude, low=None, high=None):
    """
    创建频率掩码，过滤掉低于 low Hz 和高于 high Hz 的频率
    """
    if not low or not high:
        logger.warning("No frequency range specified, returning all frequencies.")
        return freqs, magnitude
    mask = (freqs >= low) & (freqs <= high)
    return freqs[mask], magnitude[mask]


def plot_freq(data1, data2, save_path=None):
    """
    创建频率域分析图
    """
    logger.info("Creating frequency analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold')
    
    # Lab 数据频谱分析
    lab_accel = np.array(data1['accel'])
    # lab_fft = np.fft.fft(lab_accel[:min(len(lab_accel), LAB_SAMPLING_RATE*5)])  # 分析前5秒
    lab_fft = np.fft.fft(lab_accel)  # 分析全部数据
    lab_freqs = np.fft.fftfreq(len(lab_fft), 1/LAB_SAMPLING_RATE)
    lab_magnitude = np.abs(lab_fft)
    # 可选：过滤到0-10Hz范围
    lab_freqs_filtered, lab_magnitude_filtered = special_filter_mask(lab_freqs, lab_magnitude)

    # Aw 数据频谱分析
    ax_accel = np.array(data2['accel'])
    # ax_fft = np.fft.fft(ax_accel[:min(len(ax_accel), AW_SAMPLING_RATE*5)])  # 分析前5秒
    ax_fft = np.fft.fft(ax_accel)  # 分析全部数据
    ax_freqs = np.fft.fftfreq(len(ax_fft), 1/AW_SAMPLING_RATE)
    ax_magnitude = np.abs(ax_fft)
    # 可选：过滤到0-10Hz范围
    ax_freqs_filtered, ax_magnitude_filtered = special_filter_mask(ax_freqs, ax_magnitude)

    # 绘制频谱
    axes[0].semilogy(lab_freqs_filtered, lab_magnitude_filtered)
    axes[0].set_title('Lab Data Frequency Spectrum')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_xlim(0, None)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(ax_freqs_filtered, ax_magnitude_filtered)
    axes[1].set_title('Aw Data Frequency Spectrum')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_xlim(0, None)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_frequency.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Frequency plot saved to: {save_path.replace('.png', '_frequency.png')}")
    
    plt.show()

def print_stat(data1, data2):
    """
    打印数据统计信息
    """
    print("=== Data Statistics ===")
    
    # Lab 数据统计
    lab_duration = max(data1['timestamp'])
    lab_samples = len(data1['timestamp'])
    lab_actual_rate = lab_samples / lab_duration if lab_duration > 0 else 0
    
    print(f"Data 1:")
    print(f"  Duration: {lab_duration:.2f} seconds")
    print(f"  Total samples: {lab_samples}")
    print(f"  Expected sampling rate: {LAB_SAMPLING_RATE} Hz")
    print(f"  Actual sampling rate: {lab_actual_rate:.2f} Hz")
    print(f"  X-axis range: [{np.min(data1['accelX']):.3f}, {np.max(data1['accelX']):.3f}]")
    print(f"  Y-axis range: [{np.min(data1['accelY']):.3f}, {np.max(data1['accelY']):.3f}]")
    print(f"  Z-axis range: [{np.min(data1['accelZ']):.3f}, {np.max(data1['accelZ']):.3f}]")
    
    # Aw 数据统计
    ax_duration = max(data2['timestamp'])
    ax_samples = len(data2['timestamp'])
    ax_actual_rate = ax_samples / ax_duration if ax_duration > 0 else 0
    
    print(f"Data 2:")
    print(f"  Duration: {ax_duration:.2f} seconds")
    print(f"  Total samples: {ax_samples}")
    print(f"  Expected sampling rate: {AW_SAMPLING_RATE} Hz")
    print(f"  Actual sampling rate: {ax_actual_rate:.2f} Hz")
    print(f"  X-axis range: [{np.min(data2['accelX']):.3f}, {np.max(data2['accelX']):.3f}]")
    print(f"  Y-axis range: [{np.min(data2['accelY']):.3f}, {np.max(data2['accelY']):.3f}]")
    print(f"  Z-axis range: [{np.min(data2['accelZ']):.3f}, {np.max(data2['accelZ']):.3f}]")

def downsample_data(lab_data, original_freq=1500, target_freq=20):
    """
    将lab数据降采样到目标频率
    
    Args:
        lab_data: lab数据字典，包含accelX, accelY, accelZ等
        original_freq: 原始采样频率，默认1500Hz
        target_freq: 目标频率，默认20Hz (Apple Watch频率)
    
    Returns:
        降采样后的数据字典
    """
    import numpy as np
    from scipy import signal

    if target_freq >= original_freq:
        logger.warning("Target frequency is greater than or equal to original frequency. No downsampling needed.")
        return lab_data
    
    # 计算降采样比例
    downsample_factor = original_freq // target_freq
    logger.info(f"Downsampling from {original_freq}Hz to {target_freq}Hz (factor: {downsample_factor})")
    
    downsampled_data = {}
    
    # 处理加速度数据
    for key in ['accelX', 'accelY', 'accelZ']:
        if key in lab_data and isinstance(lab_data[key], (list, np.ndarray)):
            data_array = np.array(lab_data[key])
            
            # 使用scipy的decimate函数进行降采样（包含抗混叠滤波）
            try:
                downsampled = signal.decimate(data_array, downsample_factor, ftype='fir')
                downsampled_data[key] = downsampled
                logger.info(f"Downsampled {key}: {len(data_array)} -> {len(downsampled)} samples")
            except Exception as e:
                logger.warning(f"Decimate failed for {key}, using simple downsampling: {e}")
                # 如果decimate失败，使用简单的间隔采样
                downsampled_data[key] = data_array[::downsample_factor]
                logger.info(f"Simple downsampled {key}: {len(data_array)} -> {len(downsampled_data[key])} samples")
    
    # 重新计算时间戳
    if 'accelX' in downsampled_data:
        new_length = len(downsampled_data['accelX'])
        downsampled_data['timestamp'] = np.arange(new_length) / target_freq
        logger.info(f"Generated new timestamp array with {new_length} samples")
    
    # 重新计算合成加速度
    if all(key in downsampled_data for key in ['accelX', 'accelY', 'accelZ']):
        downsampled_data['accel'] = np.sqrt(
            downsampled_data['accelX']**2 + 
            downsampled_data['accelY']**2 + 
            downsampled_data['accelZ']**2
        )
        logger.info(f"Recalculated magnitude acceleration with {len(downsampled_data['accel'])} samples")
    
    return downsampled_data

def main():
    parser = argparse.ArgumentParser(description='Visualize Lab and AW data comparison')
    parser.add_argument('--lab_file', type=str, help='Path to lab data file (.joblib)')
    parser.add_argument('--aw_file', type=str, help='Path to aw data file (.csv)')
    parser.add_argument('--save_dir', type=str, default='./visualizations', 
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # 如果没有提供命令行参数，则交互式输入
    if not args.lab_file:
        args.lab_file = input("请输入 Lab 数据文件路径 (.joblib): ").strip()
        # get rid of surrounding quotes if present
        if args.lab_file.startswith('"') and args.lab_file.endswith('"'):
            args.lab_file = args.lab_file[1:-1]
        elif args.lab_file.startswith("'") and args.lab_file.endswith("'"):
            args.lab_file = args.lab_file[1:-1]

    if not args.aw_file:
        args.aw_file = input("请输入 AW 数据文件路径 (.csv): ").strip()
        if args.aw_file.startswith('"') and args.aw_file.endswith('"'):
            args.aw_file = args.aw_file[1:-1]
        elif args.aw_file.startswith("'") and args.aw_file.endswith("'"):
            args.aw_file = args.aw_file[1:-1]

    try:
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        
        logger.info("=== Starting Data Processing Pipeline ===")
        
        # 加载数据
        logger.info("Loading Lab data...")
        lab_data = load_lab_file(args.lab_file)
        
        logger.info("Loading Apple Watch data...")
        aw_data = load_aw_file(args.aw_file)  # 使用默认的17-35秒裁剪
        
        # 对Lab数据进行降采样到20Hz（所有采样率配置都在这里）
        logger.info("Processing Lab data...")
        lab_data_processed = downsample_data(lab_data, original_freq=1500, target_freq=20)

        # 打印统计信息
        logger.info("=== Data Statistics ===")
        print_stat(lab_data_processed, aw_data)

        # 生成文件名
        lab_filename = Path(args.lab_file).stem
        aw_filename = Path(args.aw_file).stem
        save_path = os.path.join(args.save_dir, f"{lab_filename}_vs_{aw_filename}_comparison.png")

        # 创建可视化（所有绘图配置都在各个函数中）
        logger.info("Creating visualizations...")
        plot_comparison(lab_data_processed, aw_data, save_path)
        
        # 可选：启用频域分析
        # plot_freq(lab_data_processed, aw_data, save_path)

        logger.success("Visualization pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
