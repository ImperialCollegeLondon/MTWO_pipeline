import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AW_SAMPLING_RATE = 20  # Apple Watch sampling rate in Hz
LAB_SAMPLING_RATE = 1500  # Vicon Lab sampling rate in Hz

def load_csv_data(csv_path: str, data_type: str) -> pd.DataFrame:
    """
    Load calibration data from a CSV file.
    
    @param csv_path: Path to the CSV file containing calibration data.
    @param data_type: Type of data, either 'aw' for Apple Watch or 'lab' for Vicon Lab.
    @return: DataFrame with columns 'accelX', 'accelY', 'accelZ', and 'accel'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Please check path: {csv_path}")
    if data_type.lower() not in ['aw', 'lab']:
        raise ValueError("data_type must be either 'aw' or 'lab'.")
    
    data = pd.read_csv(csv_path)

    # determine timestamp
    timestamp_col = None
    for col in data.columns:
        if 'timestamp' in col.lower() or 'time' in col.lower():
            timestamp_col = col
            break
    if not timestamp_col:
        sampling_rate = AW_SAMPLING_RATE if data_type.lower()=='aw' else LAB_SAMPLING_RATE
        data['Timestamp'] = pd.Series(range(len(data))) / sampling_rate
        timestamp_col = 'Timestamp'
    # determine acceleration for each axis
    for col in data.columns:
        col_lower = col.lower()
        if 'accelx' in col_lower or 'accelerationx' in col_lower:
            accel_x_col = col
        elif 'accely' in col_lower or 'accelerationy' in col_lower:
            accel_y_col = col
        elif 'accelz' in col_lower or 'accelerationz' in col_lower:
            accel_z_col = col


    df = data[[timestamp_col, accel_x_col, accel_y_col, accel_z_col]].copy()
    df.columns = ["Timestamp", "AccelX", "AccelY", "AccelZ"]
    
    df['Accel'] = np.sqrt(df['AccelX']**2 + df['AccelY']**2 + df['AccelZ']**2)

    return pd.DataFrame({
        'timestamp': df['Timestamp'].values,
        'accelX': df['AccelX'].values,
        'accelY': df['AccelY'].values,
        'accelZ': df['AccelZ'].values,
        'accel': df['Accel'].values
    })

def plot_original(data, save_path=None):
    fig, axes = plt.subplots(2,1, figsize=(8, 5))
    fig.suptitle('Acceleration Data', fontsize=16)

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
        print(f"Plot saved to: {save_path}")

    plt.show()

def compare(aw_data, vicon_data):
    fig, axes = plt.subplots(2,1, figsize=(8, 5))
    fig.suptitle('Acceleration Comparison', fontsize=16)

    axes[0].plot(aw_data['timestamp'], aw_data['accelX'], linewidth=1, color='red', label='X')
    axes[0].plot(aw_data['timestamp'], aw_data['accelY'], linewidth=1, color='green', label='Y')
    axes[0].plot(aw_data['timestamp'], aw_data['accelZ'], linewidth=1, color='blue', label='Z')
    axes[0].set_title('Apple Watch Acceleration')
    axes[0].set_ylabel('Acceleration (m/s^2)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1, 1))

    axes[1].plot(vicon_data['timestamp'], vicon_data['accelX'], linewidth=1, color='red', label='X')
    axes[1].plot(vicon_data['timestamp'], vicon_data['accelY'], linewidth=1, color='green', label='Y')
    axes[1].plot(vicon_data['timestamp'], vicon_data['accelZ'], linewidth=1, color='blue', label='Z')
    axes[1].set_title('Vicon Acceleration')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Acceleration (m/s^2)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

def plot_chopped(data, chopped_range, save_path=None):
    fig, axes = plt.subplots(2,1, figsize=(8, 5))
    fig.suptitle('Acceleration Data before and after chopping', fontsize=16)

    # Original acceleration data
    axes[0].plot(data['timestamp'], data['accelX'], linewidth=1, color='red', label='X')
    axes[0].plot(data['timestamp'], data['accelY'], linewidth=1, color='green', label='Y')
    axes[0].plot(data['timestamp'], data['accelZ'], linewidth=1, color='blue', label='Z')
    
    # Highlight the chopped range with bold line
    axes[0].plot(data['timestamp'][chopped_range], data['accelX'][chopped_range], linewidth=2, color='darkred')
    axes[0].plot(data['timestamp'][chopped_range], data['accelY'][chopped_range], linewidth=2, color='darkgreen')
    axes[0].plot(data['timestamp'][chopped_range], data['accelZ'][chopped_range], linewidth=2, color='darkblue')

    axes[0].set_title('Original Acceleration')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Acceleration (m/s^2)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Chopped acceleration data
    axes[1].plot(data['timestamp'][chopped_range], data['accelX'][chopped_range], linewidth=1, color='red', label='X')
    axes[1].plot(data['timestamp'][chopped_range], data['accelY'][chopped_range], linewidth=1, color='green', label='Y')
    axes[1].plot(data['timestamp'][chopped_range], data['accelZ'][chopped_range], linewidth=1, color='blue', label='Z')
    axes[1].set_title('Chopped Acceleration')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Acceleration (m/s^2)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    """ 
    csv_path = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\Calibration Data\AppleWatch\M1-S0077.csv"
    aw = load_csv_data(csv_path, 'aw')

    csv_path = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\Calibration Data\Lab\LabCal\chopped_right_M2TestingWalking01.csv"
    vicon = load_csv_data(csv_path, 'lab')

    compare(aw, vicon)
    """
    data = load_csv_data(r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\my_data\GERF-R-D820-M3-S0016.csv", 'aw')
    plot_original(data)