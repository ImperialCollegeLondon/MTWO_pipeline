import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import datetime
from pathlib import Path
from loguru import logger
from scipy import signal
from scipy.fft import fft, fftfreq
from dataTransformer.filter import filter

logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

AW_SAMPLING_RATE = 20

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_apple_watch_data(file_path):
    logger.info(f"Loading Apple Watch data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    

    data = pd.read_csv(file_path)
    
    timestamp_col = None
    accel_x_col = None
    accel_y_col = None
    accel_z_col = None
    
    for col in data.columns:
        if 'timestamp' in col.lower() or 'time' in col.lower():
            timestamp_col = col
            break
    for col in data.columns:
        col_lower = col.lower()
        if 'accelerationx' in col_lower:
            accel_x_col = col
        elif 'accelerationy' in col_lower:
            accel_y_col = col
        elif 'accelerationz' in col_lower:
            accel_z_col = col
    
    
    logger.info(f"Found columns - Timestamp: {timestamp_col}, X: {accel_x_col}, Y: {accel_y_col}, Z: {accel_z_col}")
    

    subset = data[[timestamp_col, accel_x_col, accel_y_col, accel_z_col]].copy()
    subset.columns = ["Timestamp", "X", "Y", "Z"]
    

    if subset["Timestamp"].dtype == 'object':
        try:
            subset["Timestamp"] = pd.to_datetime(subset["Timestamp"])
        except:
            subset["Timestamp"] = subset["Timestamp"].map(
                lambda t: datetime.datetime.fromtimestamp(t) if isinstance(t, (int, float)) else t
            )
    elif subset["Timestamp"].dtype in ['int64', 'float64']:
        subset["Timestamp"] = subset["Timestamp"].map(
            lambda t: datetime.datetime.fromtimestamp(t)
        )
    
    start_time = subset["Timestamp"].iloc[0]
    subset["RelativeTime"] = (subset["Timestamp"] - start_time).dt.total_seconds()
    
    logger.info("Applying Butterworth filter to Apple Watch data...")
    accel_x_filtered = filter(subset["X"].values)
    accel_y_filtered = filter(subset["Y"].values)
    accel_z_filtered = filter(subset["Z"].values)
    
    accel_magnitude = np.sqrt(accel_x_filtered**2 + accel_y_filtered**2 + accel_z_filtered**2)
    
    processed_data = {
        'timestamp': subset["RelativeTime"].values,
        'accelX': accel_x_filtered,
        'accelY': accel_y_filtered,
        'accelZ': accel_z_filtered,
        'accel': accel_magnitude,
        'raw_timestamp': subset["Timestamp"].values
    }
    
    logger.info(f"Loaded {len(processed_data['timestamp'])} samples, duration: {processed_data['timestamp'][-1]:.2f}s")
    
    return processed_data

def plot_time_series(data, save_path=None): 
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Apple Watch Acceleration Data - Time Series', fontsize=16, fontweight='bold')
    
    # X acceleration
    axes[0, 0].plot(data['timestamp'], data['accelX'], linewidth=1)
    axes[0, 0].set_title('X-axis Acceleration')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Acceleration (g)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Y acceleration
    axes[0, 1].plot(data['timestamp'], data['accelY'], linewidth=1, color='orange')
    axes[0, 1].set_title('Y-axis Acceleration')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Acceleration (g)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Z acceleration
    axes[1, 0].plot(data['timestamp'], data['accelZ'], linewidth=1, color='green')
    axes[1, 0].set_title('Z-axis Acceleration')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Acceleration (g)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Magnitude acceleration
    axes[1, 1].plot(data['timestamp'], data['accel'], linewidth=1, color='red')
    axes[1, 1].set_title('Magnitude Acceleration')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Acceleration (g)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_timeseries.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Time series plot saved to: {save_path.replace('.png', '_timeseries.png')}")
    
    plt.show()

def print_data_summary(data):
    print("\n=== Apple Watch Data Summary ===")
    
    duration = data['timestamp'][-1] - data['timestamp'][0]
    samples = len(data['timestamp'])
    sampling_rate = samples / duration if duration > 0 else 0
    
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total samples: {samples}")
    print(f"Sampling rate: {sampling_rate:.2f} Hz")
    print(f"Expected sampling rate: {AW_SAMPLING_RATE} Hz")
    
    print("\n=== Statistical Summary ===")
    for axis, values in [('X', data['accelX']), ('Y', data['accelY']), ('Z', data['accelZ']), ('Magnitude', data['accel'])]:
        print(f"{axis}-axis:")
        print(f"  Mean: {np.mean(values):.4f} g")
        print(f"  Std: {np.std(values):.4f} g")
        print(f"  Min: {np.min(values):.4f} g")
        print(f"  Max: {np.max(values):.4f} g")
        print(f"  Range: {np.max(values) - np.min(values):.4f} g")

def main():
    parser = argparse.ArgumentParser(description='Apple Watch Data Visualization Tool')
    parser.add_argument('--file', type=str, help='Path to Apple Watch data file (.csv)')
    parser.add_argument('--start_time', type=float, help='Start time for cropping (seconds)')
    parser.add_argument('--end_time', type=float, help='End time for cropping (seconds)')
    parser.add_argument('--save_dir', type=str, default='/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/my_data/visualization', 
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    if not args.file:
        args.file = input("Enter Apple Watch Path(.csv):")
        # 如何路径首尾存在双/单引号则去除
        if args.file.startswith(("'", '"')) and args.file.endswith(("'", '"')):
            args.file = args.file[1:-1]

    try:
        os.makedirs(args.save_dir, exist_ok=True)

        data = load_apple_watch_data(args.file)
        
        print_data_summary(data)
        
        filename = Path(args.file).stem
        save_path = os.path.join(args.save_dir, f"{filename}_visualization.png")
        
        plot_time_series(data, save_path)
        
        logger.success("Apple Watch data visualization completed!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
