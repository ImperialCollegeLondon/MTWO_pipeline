import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import datetime
from pathlib import Path
from loguru import logger
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np
from scipy.signal import butter, lfilter
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

def filter(data, order=4, cutoff_freq=5, sampling_rate=200):
    '''The meaning of the parameters of butter:
    N=4 means the order of the filter
    Wn=5 means the cutoff frequency is 5Hz.
    fs=200 means the sampling frequency is 200Hz.'''
    b, a = butter(order, [cutoff_freq], fs=sampling_rate)
    filtered_data = lfilter(b, a, data)
    return filtered_data

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
        if 'accelx' in col_lower or 'accelerationx' in col_lower:
            accel_x_col = col
        elif 'accely' in col_lower or 'accelerationy' in col_lower:
            accel_y_col = col
        elif 'accelz' in col_lower or 'accelerationz' in col_lower:
            accel_z_col = col
    if not timestamp_col:
        data['Timestamp'] = pd.Series(range(len(data))) / AW_SAMPLING_RATE
        timestamp_col = 'Timestamp'
    
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
    
    accel_x_filtered = filter(subset["X"].values)
    accel_y_filtered = filter(subset["Y"].values)
    accel_z_filtered = filter(subset["Z"].values)
    # accel_x_filtered = subset["X"].values
    # accel_y_filtered = subset["Y"].values
    # accel_z_filtered = subset["Z"].values
    
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
    fig, axes = plt.subplots(2,1, figsize=(8, 5))
    fig.suptitle('Apple Watch Acceleration Data - Time Series', fontsize=16)

    axes[0].plot(data['timestamp'], data['accelX'], linewidth=1, color='red', label='X')
    axes[0].plot(data['timestamp'], data['accelY'], linewidth=1, color='green', label='Y')
    axes[0].plot(data['timestamp'], data['accelZ'], linewidth=1, color='blue', label='Z')
    axes[0].set_title('Axis Acceleration')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Magnitude acceleration
    axes[1].plot(data['timestamp'], data['accel'], linewidth=1, color='black')
    axes[1].set_title('Magnitude Acceleration')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Acceleration (g)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.show()

def print_data_summary(data):
    try:
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
    except Exception as e:
        logger.error(f"Error printing data summary: {e}")
        print("Error printing data summary. Please check the data format and content.")

def main():
    path = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/Calibration Data/AppleWatch/AW_Chopped/chopped_M2-S0079.csv'
    # path = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/Calibration Data/Lab/LabChopped/chopped_right_M2TestingDrinking03.csv'
    start_time = None
    end_time = 275
    parser = argparse.ArgumentParser(description='Visualization Tool')
    parser.add_argument('--file', default=path)
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


    os.makedirs(args.save_dir, exist_ok=True)
    data = load_apple_watch_data(args.file)

    if args.start_time is not None:
        start_time = args.start_time
    else:
        start_time = data['timestamp'][0]
    if args.end_time is not None:
        end_time = args.end_time
    else:
        end_time = data['timestamp'][-1]
    # Crop data based on start and end time
    data = {
        'timestamp': data['timestamp'][(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)],
        'accelX': data['accelX'][(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)],
        'accelY': data['accelY'][(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)],
        'accelZ': data['accelZ'][(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)],
        'accel': data['accel'][(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)],
        'raw_timestamp': data['raw_timestamp'][(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
    }
    
    print_data_summary(data)

    filename = Path(args.file).stem
    save_path = os.path.join(args.save_dir, f"{filename}.png")
    
    plot_time_series(data, save_path)
    
    logger.success("Apple Watch data visualization completed!")


if __name__ == "__main__":
    main()