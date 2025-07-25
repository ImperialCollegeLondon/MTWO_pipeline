import os
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.join(script_dir, os.pardir)
sys.path.insert(0, parent_dir)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
from loguru import logger
from dataTransformer.filter import filter
from config import getLogger
from scipy.signal import resample, butter, filtfilt, correlate
from scipy.spatial import procrustes
from dtw import dtw
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.transform import Rotation as R_scipy
logger = getLogger('INFO')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_lab_file(lab_file_path: str, side: str = 'right') -> pd.DataFrame:
    '''
    @param lab_file_path: Path to the lab data file (.joblib)
    @param side: 'left' or 'right', specifies which leg's data to load
    @return: DataFrame containing acceleration data with keys 'accelX', 'accelY', 'accelZ', and 'accel'
    '''
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

    return pd.DataFrame(accel_data)

def load_aw_file(aw_file_path: str) -> pd.DataFrame:
    ''' 
    @param aw_file_path: Path to the Apple Watch data file (.csv)
    @return: DataFrame containing acceleration data with columns 'accelX', 'accelY', 'accelZ', and 'accel'
    '''
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

    return pd.DataFrame(accel_data)

def load_calib_lab(csv_path: str) -> pd.DataFrame:
    """
    Load calibration data from a CSV file.
    
    @param csv_path: Path to the CSV file containing calibration data.
    @return: DataFrame with columns 'accelX', 'accelY', 'accelZ', and 'accel'.
    """
    logger.info(f"Loading calibration data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Calibration CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['Accel'] = np.sqrt(df['AccelX']**2 + df['AccelY']**2 + df['AccelZ']**2)

    # csv中没有timestamp，需要手动添加
    df['timestamp'] = np.arange(len(df)) / LAB_SAMPLING_RATE

    return pd.DataFrame({
        'timestamp': df['timestamp'].values,
        'accelX': df['AccelX'].values,
        'accelY': df['AccelY'].values,
        'accelZ': df['AccelZ'].values,
        'accel': df['Accel'].values
    })

def visualise(data, show=True):
    plt.plot(data['timestamp'], data['accelX'], label='X-axis')
    plt.plot(data['timestamp'], data['accelY'], label='Y-axis')
    plt.plot(data['timestamp'], data['accelZ'], label='Z-axis')
    plt.title("Acceleration Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid()
    if show:
        plt.show()

# ---------------------------------------
AW_SAMPLING_RATE = 20 # Hz, Apple Watch sampling rate
LAB_SAMPLING_RATE = 1500 # Hz, Vicon original sampling rate
CUTOFF_FREQ = 5 # Hz, Low-pass filter cutoff frequency
# ---------------------------------------

# --- 1. Data Preprocessing ---
def preprocess_acceleration_data(df: pd.DataFrame, original_sampling_rate: int, target_sampling_rate: int, cutoff_freq: float) -> pd.DataFrame:
    """
    Preprocesses the acceleration data by downsampling and applying a high-pass filter.

    @param df: DataFrame with columns 'timestamp', 'accelX', 'accelY', 'accelZ'
    @param original_sampling_rate: Original sampling rate of the data
    @param target_sampling_rate: Target sampling rate after downsampling
    @param cutoff_freq: Cutoff frequency for the low-pass filter
    @return: Preprocessed DataFrame with downsampled and filtered acceleration data
    """
    # 1.1 Downsampling
    num_samples_target = int(len(df) * (target_sampling_rate / original_sampling_rate))

    accel_data = df[['accelX', 'accelY', 'accelZ']].values
    timestamps_original = df['timestamp'].values

    resampled_accel = np.zeros((num_samples_target, 3))
    resampled_timestamps = np.linspace(timestamps_original.min(), timestamps_original.max(), num_samples_target)
    for i in range(3):
        resampled_accel[:, i] = resample(accel_data[:, i], num_samples_target)

    resampled_df = pd.DataFrame(resampled_accel, columns=['accelX', 'accelY', 'accelZ'])
    resampled_df['timestamp'] = resampled_timestamps

    # 另一种采样方法：先将Timestamp设为索引，然后使用df.resample
    # (a). Set Timestamp column as TimedeltaIndex
    # df['Timestamp'] = pd.to_timedelta(df['Timestamp'], unit='s')
    # df = df.set_index('Timestamp')

    # (b). Calculate resampling frequency
    # resample_rule = f"{1000 / target_sampling_rate:.0f}ms"

    # (c). Use resample to downsample and aggregate using mean
    # resampled_df = df.resample(resample_rule).mean().dropna()

    # (d). Reset index to make Timestamp a column again
    # resampled_df = resampled_df.reset_index()

    # 1.2 Highpass Filter
    nyquist_freq = 0.5 * target_sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(4, normal_cutoff, btype='highpass', analog=False) # 4th order Butterworth filter

    filtered_df = resampled_df.copy()
    for col in ['accelX', 'accelY', 'accelZ']:
        filtered_df[col] = filtfilt(b, a, resampled_df[col].values)
        
    return filtered_df

def normalize_data(df: pd.DataFrame) -> tuple:
    """
    Applies Z-score normalization to the 'accelX', 'accelY', and 'accelZ' columns.

    @param df: DataFrame with columns 'accelX', 'accelY', 'accelZ'
    @return: Normalized DataFrame and the scaler used for normalization
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[['accelX', 'accelY', 'accelZ']])
    normalized_df = df.copy()
    normalized_df[['accelX', 'accelY', 'accelZ']] = normalized_data
    # Save the scaler for later use
    joblib.dump(scaler, 'scaler.joblib')
    return normalized_df, scaler # Return the scaler to apply the same normalization to new data

# --- 2. Coordinate System Alignment Functions ---
def align_coordinate_systems_procrustes(source_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple:
    """
    使用Procrustes分析对齐两个数据集的坐标系。
    假设两个DataFrame已经时间同步并包含了相同的运动阶段，即source_df和target_df是已经时间对齐、且点数相同的子段
    
    Procrustes分析旨在找到最佳的旋转、缩放和平移，以最小化两个形状之间的差异。
    对于加速度数据，通常我们只关心旋转。这是因为旋转可以将一个设备的坐标系对齐到另一个设备的坐标系上，或者是一个统一的解剖坐标系。

    返回旋转后的source_df的加速度数据。
    @param source_df: DataFrame containing source acceleration data with columns 'accelX', 'accelY', 'accelZ'
    @param target_df: DataFrame containing target acceleration data with columns 'accelX', 'accelY', 'accelZ'
    @return: DataFrame with aligned source_df acceleration data and the rotation matrix
    """
    # Extract X,Y,Z columns as point sets
    source_points = source_df[['accelX', 'accelY', 'accelZ']].values
    target_points = target_df[['accelX', 'accelY', 'accelZ']].values
    
    # Procrustes分析会返回 M1_new, M2_new (对齐后的点), disparity
    # M1_new 是 M1 经过旋转、缩放、平移后的结果
    # 这里我们只关注旋转部分，scipy的procrustes不直接返回旋转矩阵，
    # 我们可以通过SVD手动计算旋转矩阵，或利用其结果。
    
    # 假设我们只需要旋转，且没有缩放和平移（即单位矩阵，中心化后）
    # 更直接的方法是基于静态校准点来计算旋转矩阵
    
    # 简化的Procrustes应用：对两个信号进行对齐
    # 注意：Procrustes更适用于形状对齐，对于时间序列的逐点对齐，需要确保点数和顺序一致
    # 在实际应用中，你可能需要用校准姿态的静态数据来计算旋转矩阵
    
    # 假设这里source_df和target_df是已经时间对齐、且点数相同的子段
    # Center the data
    source_centered = source_points - source_points.mean(axis=0)
    target_centered = target_points - target_points.mean(axis=0)
    
    # SVD to find optimal rotation
    U, s, Vt = np.linalg.svd(target_centered.T @ source_centered)
    R = U @ Vt
    
    # 如果检测到反射，调整SVD结果（Procrustes可能不处理反射）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # 应用旋转到原始source数据（未中心化）
    aligned_source_accel = (source_points @ R.T) # 注意矩阵乘法顺序
    
    aligned_df = source_df.copy()
    aligned_df[['accelX', 'accelY', 'accelZ']] = aligned_source_accel
    return aligned_df, R # 返回旋转后的数据和旋转矩阵

def calculate_rotation_matrix_from_flat_pose(vicon_flat_df, aw_flat_df):
    """
    根据平躺姿态数据计算从Apple Watch到Vicon的旋转矩阵。
    假设在平躺姿态下，两个设备都静止，且各自的Z轴都近似指向（或反向指向）重力方向。
    更通用的方法是，两个设备的Y轴（通常是设备的前向）都指向某个共同的固定方向（例如受试者头部）。
    
    这里我们使用重力向量作为参考：
    - Vicon的平均重力向量 (v_ref)
    - Apple Watch的平均重力向量 (aw_vec)
    我们将计算一个旋转矩阵，使得 aw_vec 旋转后与 v_ref 方向一致。
    
    更精确的校准会使用多于一个姿态，但平躺是最简单的单姿态校准。
    
    返回一个3x3的旋转矩阵 R，使得 `R @ aw_accel` 能够对齐到 vicon_accel 的坐标系。
    """
    # 确保数据经过预处理且干净
    vicon_accel_flat = vicon_flat_df[['accelX', 'accelY', 'accelZ']].values
    aw_accel_flat = aw_flat_df[['accelX', 'accelY', 'accelZ']].values

    # 计算平均加速度向量 (代表重力方向在各自坐标系下的投影)
    v_ref_vec = np.mean(vicon_accel_flat, axis=0)
    aw_current_vec = np.mean(aw_accel_flat, axis=0)

    # 归一化向量 (只关心方向)
    v_ref_vec = v_ref_vec / np.linalg.norm(v_ref_vec)
    aw_current_vec = aw_current_vec / np.linalg.norm(aw_current_vec)

    # 计算旋转矩阵（将 aw_current_vec 旋转到 v_ref_vec）
    # 使用 scipy.spatial.transform.Rotation.align_vectors
    # 它能找到将一组向量旋转到另一组向量的最佳旋转
    rot_obj, _ = R_scipy.align_vectors([v_ref_vec], [aw_current_vec])
    rotation_matrix = rot_obj.as_matrix()
    
    return rotation_matrix

def apply_rotation_matrix(df, rotation_matrix):
    """
    将计算出的旋转矩阵应用到DataFrame的加速度数据上。
    """
    accel_data = df[['accelX', 'accelY', 'accelZ']].values
    rotated_accel = (rotation_matrix @ accel_data.T).T # 注意矩阵乘法顺序
    
    rotated_df = df.copy()
    rotated_df[['accelX', 'accelY', 'accelZ']] = rotated_accel
    return rotated_df

def mirror_data_for_hand(df: pd.DataFrame, hand_type: str = 'left') -> pd.DataFrame:
    """
    根据手部佩戴类型，对数据进行镜像处理，以统一左右手的方向。
    假设一个标准解剖坐标系：X=内外侧, Y=前后, Z=上下。
    左手与右手在X轴（内外侧）上通常是镜像关系。

    @param df: DataFrame containing acceleration data with columns 'accelX', 'accelY', 'accelZ'
    @param hand_type: 当前数据的类型。'left' 或 'right'。
    @return: Mirrored DataFrame with adjusted 'X' values
    """
    mirrored_df = df.copy()
    if hand_type == 'left':
        # 假设我们想要将左手数据转换为右手视角
        # 这意味着X轴（内外侧）需要翻转
        mirrored_df['X'] = -mirrored_df['X']
        # 其他轴可能也需要根据实际设备和佩戴方式调整
    elif hand_type == 'right':
        # 如果需要将右手数据转换为左手视角，则也翻转X
        # mirrored_df['X'] = -mirrored_df['X']
        pass # 默认不处理右手
    return mirrored_df


# --- 3. Time Series Alignment ---
def initial_time_alignment(ref_df: pd.DataFrame, other_df: pd.DataFrame, column: str = 'Z') -> float:
    """
    使用交叉相关进行粗略时间对齐。
    返回other_df相对于ref_df的时间偏移（秒）。
    """
    ref_signal = ref_df[column].values
    other_signal = other_df[column].values

    # 计算交叉相关
    correlation = correlate(ref_signal, other_signal, mode='full')
    
    # 找到最大相关性的滞后量
    lags = np.arange(-(len(other_signal) - 1), len(ref_signal))
    lag = lags[np.argmax(correlation)]
    
    # 将滞后量转换为时间（以秒为单位）
    time_offset = lag / AW_SAMPLING_RATE
    
    print(f"Initial time offset (seconds): {time_offset}")
    
    # 根据offset调整other_df的时间戳
    # 这里我们不是直接修改other_df，而是返回offset供后续切片
    return time_offset

def align_with_dtw(series1: np.ndarray, series2: np.ndarray):
    """
    使用DTW对齐两个时间序列。
    输入为NumPy数组 (X, Y, Z)。
    @return: DTW对齐结果，包含扭曲路径和距离。使用方式：alignment.index1 和 alignment.index2
    """
    dist_method = lambda x, y: np.linalg.norm(x - y) # 欧氏距离作为距离度量
    alignment = dtw(series1, series2, dist_method)
    
    # alignment.index1 和 alignment.index2 提供了扭曲路径
    # 这些索引可以用来重新采样或对齐两个序列
    return alignment

# --- 4. 映射模型训练函数 ---
def train_mapping_model(aligned_vicon_accel: pd.DataFrame, aligned_aw_accel: pd.DataFrame) -> LinearRegression:
    """
    训练一个线性回归模型，将Vicon加速度映射到Apple Watch加速度。
    """
    # 将输入数据转换为 (n_samples, n_features) 形状
    X = aligned_vicon_accel[['accelX', 'accelY', 'accelZ']].values
    y = aligned_aw_accel[['accelX', 'accelY', 'accelZ']].values # 目标是Apple Watch的XYZ

    # 可以针对每个轴训练一个模型，或者训练一个多输出模型
    model = LinearRegression()
    model.fit(X, y)
    
    # 评估模型
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"Mapping Model Training Result:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    return model


if __name__ == "__main__":
    '''
    # --- 0. 加载平躺姿态数据 ---
    print("--- 0. 加载平躺姿态数据 ---")
    root_dir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/lying_data/'
    flat_vicon_filepath = root_dir + 'L05 Lying01.c3d.ontrackclassifier.joblib'
    vicon_flat_data_raw = load_lab_file(flat_vicon_filepath, side='right')
    flat_aw_filepath = root_dir + 'GERF-L-D524-M3-S0041.csv' # self collected Apple Watch data, with the same lying pose as vicon.
    aw_flat_data_raw = load_aw_file(flat_aw_filepath)

    # 可视化平躺姿态数据，并根据可视化结果粗略截取apple watch数据，使得两者的时间长度大体一致
    # visualise(vicon_flat_data_raw)
    # visualise(aw_flat_data_raw)
    # exit()

    # 手动截取
    vicon_flat_data_raw = vicon_flat_data_raw[(vicon_flat_data_raw['timestamp'] >= 2) & (vicon_flat_data_raw['timestamp'] <= 39)]
    vicon_flat_data_raw['timestamp'] -= vicon_flat_data_raw['timestamp'].min() # 将vicon的时间戳转换为相对时间（从0开始）
    # 将aw的所有加速度从g转化为m/s^2
    aw_flat_data_raw['accelX'] *= 9.81
    aw_flat_data_raw['accelY'] *= 9.81
    aw_flat_data_raw['accelZ'] *= 9.81
    aw_flat_data_raw = aw_flat_data_raw[(aw_flat_data_raw['timestamp'] >= 122) & (aw_flat_data_raw['timestamp'] <= 159)]
    aw_flat_data_raw['timestamp'] -= aw_flat_data_raw['timestamp'].min() # 将aw的时间戳转换为相对时间（从0开始）

    # 对平躺姿态数据进行预处理（降采样到目标采样率，并滤波）
    vicon_flat_processed = preprocess_acceleration_data(
        vicon_flat_data_raw, LAB_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ
    )
    aw_flat_processed = preprocess_acceleration_data(
        aw_flat_data_raw, AW_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ
    )

    # plt.subplot(2, 1, 1)
    # visualise(vicon_flat_processed, show=False)
    # plt.subplot(2, 1, 2)
    # visualise(aw_flat_processed, show=True)

    # --- 1. 计算坐标系旋转矩阵 (基于平躺姿态) ---
    # Calculate the rotation matrix to align Apple Watch to Vicon
    print("\n--- 1. 计算坐标系旋转矩阵 ---")
    # 假设 Apple Watch 是源设备，Vicon 是目标设备
    rotation_matrix_aw_to_vicon = calculate_rotation_matrix_from_flat_pose(
        vicon_flat_processed, aw_flat_processed
    )
    print("旋转矩阵计算完成:", rotation_matrix_aw_to_vicon)

    exit()
    '''
    # --- 1. 定义坐标系旋转矩阵 ---
    # 这里的旋转矩阵已经由上面的代码基于平躺姿态数据计算得到
    rotation_matrix_aw_to_vicon = np.array([[ 0.92497707, -0.34169976,  0.16630901],
                                            [ 0.25977493,  0.24911404, -0.93298402],
                                            [ 0.27737051,  0.90619174,  0.31918981]])

    
    # --- 2. 加载运动数据 ---
    calib_data_dir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/Calibration Data/'
    motion_vicon_filepath = os.path.join(calib_data_dir, 'Lab/LabChopped/chopped_right_M2TestingDrinking03.csv')
    motion_aw_filepath = os.path.join(calib_data_dir, 'AppleWatch/AW_Chopped/chopped_M2-S0079.csv')
    vicon_motion_data_raw = load_calib_lab(motion_vicon_filepath)
    aw_motion_data_raw = load_aw_file(motion_aw_filepath)

    print(f"Vicon 运动数据点数: {len(vicon_motion_data_raw)}")
    print(f"Apple Watch 运动数据点数: {len(aw_motion_data_raw)}")
    
    # --- 3. 预处理运动数据 ---
    vicon_motion_processed = preprocess_acceleration_data(
        vicon_motion_data_raw, LAB_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ
    )
    aw_motion_processed = preprocess_acceleration_data(
        aw_motion_data_raw, AW_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ
    )
    
    # 对预处理后的数据进行归一化 (可选，但在某些情况下对DTW和映射有帮助)
    vicon_motion_normalized, vicon_scaler = normalize_data(vicon_motion_processed)
    aw_motion_normalized, aw_scaler = normalize_data(aw_motion_processed)
    
    print(f"Vicon 处理后数据点数 ({AW_SAMPLING_RATE}Hz): {len(vicon_motion_processed)}")
    print(f"Apple Watch 处理后数据点数 ({AW_SAMPLING_RATE}Hz): {len(aw_motion_processed)}")

    # --- 4. 应用坐标系旋转 ---
    # 将 Apple Watch 运动数据旋转到 Vicon 的坐标系
    aw_motion_rotated = apply_rotation_matrix(aw_motion_normalized, rotation_matrix_aw_to_vicon)
    vicon_motion_rotated = vicon_motion_normalized.copy()  # Vicon数据不需要旋转
    
    # 如果需要，进行左右手镜像处理 (例如，Vicon在右手，AW在左手)
    # aw_motion_rotated = mirror_data_for_hand(aw_motion_rotated, reference_hand='right', current_hand='left')


    # --- 5. 初始时间对齐 (粗略对齐) ---
    # 手动进行粗略对齐！
    valid_start, valid_end = 0, len(vicon_motion_rotated) # 暂且使用整个数据段
    vicon_accel_for_dtw = vicon_motion_rotated[['accelX', 'accelY', 'accelZ']].values[valid_start:valid_end]
    valid_start, valid_end = 0, len(aw_motion_rotated)
    aw_accel_for_dtw = aw_motion_rotated[['accelX', 'accelY', 'accelZ']].values[valid_start:valid_end]


    # --- 6. 精细时间对齐 (DTW) ---
    print("\n--- 精细时间对齐 (DTW) ---")
    # DTW 对齐坐标系后的数据
    alignment_result = align_with_dtw(vicon_accel_for_dtw, aw_accel_for_dtw)
    
    print(f"DTW 距离: {alignment_result.distance:.4f}")

    # 获取对齐后的序列（通过 DTW 路径重新采样）
    aligned_vicon_series = vicon_accel_for_dtw[alignment_result.index1]
    aligned_aw_series = aw_accel_for_dtw[alignment_result.index2]

    print(f"DTW 对齐后 Vicon 序列长度: {len(aligned_vicon_series)}")
    print(f"DTW 对齐后 Apple Watch 序列长度: {len(aligned_aw_series)}")

    # 可视化对齐效果
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axis_labels = ['X', 'Y', 'Z']
    for i, ax in enumerate(axs.flat[:3]):
        ax.plot(aligned_vicon_series[:, i], label=f'Vicon {axis_labels[i]}-axis (DTW Aligned)')
        ax.plot(aligned_aw_series[:, i], label=f'Apple Watch {axis_labels[i]}-axis (DTW Aligned)', linestyle='--')
        ax.set_title(f"{axis_labels[i]}-axis Acceleration after DTW Alignment")
        ax.set_xlabel("Aligned Sample Index")
        ax.set_ylabel("Acceleration (g)")
        ax.legend()
        ax.grid(True)
    # 合成加速度
    vicon_magnitude = np.linalg.norm(aligned_vicon_series, axis=1)
    aw_magnitude = np.linalg.norm(aligned_aw_series, axis=1)
    axs[1, 1].plot(vicon_magnitude, label='Vicon Magnitude (DTW Aligned)')
    axs[1, 1].plot(aw_magnitude, label='Apple Watch Magnitude (DTW Aligned)', linestyle='--')
    axs[1, 1].set_title("Magnitude Acceleration after DTW Alignment")
    axs[1, 1].set_xlabel("Aligned Sample Index")
    axs[1, 1].set_ylabel("Acceleration (g)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    plt.tight_layout()
    plt.show()

    # --- 7. 映射模型训练 ---
    print("\n--- 7. 映射模型训练 ---")
    # 使用DTW对齐后的数据训练映射模型
    mapping_model = train_mapping_model(pd.DataFrame(aligned_vicon_series, columns=['accelX','accelY','accelZ']), 
                                        pd.DataFrame(aligned_aw_series, columns=['accelX','accelY','accelZ']))
    
    # 保存模型
    model_save_path = 'mapping_model.joblib'
    joblib.dump(mapping_model, model_save_path)
    print(f"映射模型已保存到: {model_save_path}")
    
    exit()
    

    # --- 8. 映射模型应用示例 ---
    print("\n--- 8. 映射模型应用示例 ---")
    # 因为Vicon是高精度设备，我们假设它是“源”或“输入”，Apple Watch是“目标”或“输出”
    # 加载mapping_model
    mapping_model = joblib.load('mapping_model.joblib')
    
    # 从 aligned_vicon_series 中取一些点进行预测演示
    sample_vicon_input = load_calib_lab('/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/Calibration Data/Lab/LabChopped/chopped_right_M2TestingReading02.csv')
    sample_vicon_input = preprocess_acceleration_data(sample_vicon_input, LAB_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ)
    # normalize the sample input using the scaler
    scaler = joblib.load('scaler.joblib')
    sample_vicon_input = scaler.transform(sample_vicon_input[['accelX', 'accelY', 'accelZ']])
    sample_vicon_input = pd.DataFrame(sample_vicon_input, columns=['accelX', 'accelY', 'accelZ'])
    predicted_aw_accel = mapping_model.predict(sample_vicon_input)

    print(f"Vicon 输入:\n{sample_vicon_input}")
    print(f"预测的 Apple Watch 加速度:\n{predicted_aw_accel}")
    
    # 与真实的 Apple Watch 对齐数据进行比较
    true_aw_accel = load_aw_file('/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/Calibration Data/AppleWatch/AW_Chopped/chopped_M3-S0078.csv')
    print(f"真实的 Apple Watch 加速度:\n{true_aw_accel}")

    # 评估预测准确性
    mse_sample = mean_squared_error(true_aw_accel, predicted_aw_accel)
    print(f"示例预测的均方误差 (MSE): {mse_sample:.4f}")

    # print("\n--- 流程完成 ---")