import os
import sys
sys.path.insert(0, r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
import pickle
from loguru import logger
from dataTransformer.filter import filter
from config import getLogger
from scipy.signal import resample, butter, filtfilt, correlate
from scipy.spatial import procrustes
from dtw import dtw
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from scipy.spatial.transform import Rotation as R_scipy
logger = getLogger('INFO')

# Set plot style
# plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")
# plt.rcParams['font.family'] = 'SimHei'

def low_pass_filter(data, cutoff_freq=5, sampling_rate=20, order=4):
    """
    Apply a low-pass Butterworth filter to the data.
    
    Parameters:
    - data: Input data to be filtered.
    - cutoff_freq: Cutoff frequency for the low-pass filter (Hz).
    - sampling_rate: Sampling rate of the data (Hz).
    - order: Order of the Butterworth filter.
    
    Returns:
    - Filtered data.
    """
    b, a = butter(order, cutoff_freq / (0.5 * sampling_rate), btype='low')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def load_lab_file(lab_file_path: str, side: str = 'right') -> pd.DataFrame:
    '''
    Load Filtered Lab data from a joblib file.
    @param lab_file_path: Path to the lab data file (.joblib)
    @param side: 'left' or 'right', specifies which leg's data to load
    @return: DataFrame containing acceleration data with keys 'accelX', 'accelY', 'accelZ', and 'accel'
    '''
    logger.debug(f"Loading lab data from: {lab_file_path}")
    
    if not os.path.exists(lab_file_path):
        raise FileNotFoundError(f"Lab file not found: {lab_file_path}")

    lab_data = joblib.load(lab_file_path)
    if 'left' in lab_data and 'right' in lab_data:
        data = lab_data[side]
        logger.info(f"Using {side} leg data from lab file")
    else:
        logger.warning("Lab file does not contain left/right data, using the first one in the structure.")
        data = list(lab_data.values())[0] # 选第一个
    
    # Extract acceleration data
    accel_x = data.get('accelX', [])
    accel_y = data.get('accelY', [])
    accel_z = data.get('accelZ', [])

    assert len(accel_x) == len(accel_y) == len(accel_z), "Acceleration data arrays must have the same length."
    assert len(accel_x) > 0, "Acceleration data cannot be empty."

    logger.info(f"Applying low-pass filter to lab data with {len(accel_x)} samples.")
    accel_x_filtered = low_pass_filter(np.array(accel_x), sampling_rate=LAB_SAMPLING_RATE)
    accel_y_filtered = low_pass_filter(np.array(accel_y), sampling_rate=LAB_SAMPLING_RATE)
    accel_z_filtered = low_pass_filter(np.array(accel_z), sampling_rate=LAB_SAMPLING_RATE)

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
    Load Filtered Apple Watch data from a CSV file.
    @param aw_file_path: Path to the Apple Watch data file (.csv)
    @return: DataFrame containing acceleration data with columns 'accelX', 'accelY', 'accelZ', and 'accel'
    '''
    logger.debug(f"Loading Apple Watch data from: {aw_file_path}")
    
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
    logger.info(f"Applying low-pass filter to Apple Watch data with {len(aw_subset)} samples.")
    accel_x_filtered = low_pass_filter(aw_subset["X"].values, sampling_rate=AW_SAMPLING_RATE)
    accel_y_filtered = low_pass_filter(aw_subset["Y"].values, sampling_rate=AW_SAMPLING_RATE)
    accel_z_filtered = low_pass_filter(aw_subset["Z"].values, sampling_rate=AW_SAMPLING_RATE)
    accel_magnitude_filtered = np.sqrt(accel_x_filtered**2 + accel_y_filtered**2 + accel_z_filtered**2)

    accel_data = { 
        'timestamp': aw_subset["RelativeTime"].values,
        'accelX': accel_x_filtered,
        'accelY': accel_y_filtered,
        'accelZ': accel_z_filtered,
        'accel': accel_magnitude_filtered
    }

    return pd.DataFrame(accel_data)

def load_csv_data(csv_path: str, data_type: str) -> pd.DataFrame:
    """
    Load calibration data from a CSV file.
    
    @param csv_path: Path to the CSV file containing calibration data.
    @param data_type: Type of data, either 'aw' for Apple Watch or 'lab' for AX Lab.
    @return: DataFrame with columns 'accelX', 'accelY', 'accelZ', and 'accel'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Calibration CSV file not found: {csv_path}")
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


def load_paired_training_data(pickle_path: str) -> list:
    """
    Load paired training data from pickle file generated by build_training_set.py
    
    @param pickle_path: Path to the pickle file containing paired training samples
    @return: List of dictionaries containing paired training samples
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Paired training data file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        training_samples = pickle.load(f)
    
    logger.info(f"Loaded {len(training_samples)} paired training samples from {pickle_path}")
    return training_samples


def convert_sample_to_dataframes(sample: dict) -> tuple:
    """
    Convert a single paired sample to AW and AX DataFrames
    
    @param sample: Dictionary containing paired sample data
    @return: Tuple of (aw_df, AX_df)
    """
    aw_df = pd.DataFrame({
        'timestamp': sample['timestamp'],
        'accelX': sample['aw_accelX'],
        'accelY': sample['aw_accelY'],
        'accelZ': sample['aw_accelZ']
    })
    
    AX_df = pd.DataFrame({
        'timestamp': sample['timestamp'],
        'accelX': sample['AX_accelX'],
        'accelY': sample['AX_accelY'],
        'accelZ': sample['AX_accelZ']
    })
    
    return aw_df, AX_df


def process_paired_samples_with_alignment(training_samples: list, alignment_method: str = 'rotation_matrix') -> tuple:
    """
    Process all paired samples with coordinate system alignment
    
    @param training_samples: List of paired training samples
    @param alignment_method: Method for alignment ('rotation_matrix', 'procrustes', or 'none')
    @return: Tuple of (aligned_aw_data_list, aligned_AX_data_list, alignment_info)
    """
    aligned_aw_data = []
    aligned_AX_data = []
    alignment_info = {
        'method': alignment_method,
        'rotation_matrices': [],
        'sample_stats': []
    }
    
    logger.info(f"Processing {len(training_samples)} paired samples with alignment method: {alignment_method}")
    
    for i, sample in enumerate(training_samples):
        try:
            # Convert sample to DataFrames
            aw_df, AX_df = convert_sample_to_dataframes(sample)
            
            # Apply coordinate system alignment
            if alignment_method == 'rotation_matrix':
                # Calculate rotation matrix based on mean gravity vectors
                # rotation_matrix = calculate_rotation_matrix_from_sample(AX_df, aw_df)
                rotation_matrix = np.array([[ 0.58377147,  0.28237329,  0.76123334],
                                            [ 0.36897715,  0.74289845, -0.55853179],
                                            [-0.72323352,  0.60693263,  0.32949362]])

                aligned_AX_df = apply_rotation_matrix(AX_df, rotation_matrix)
                aligned_aw_df = aw_df.copy()  # AW data remains unchanged
                alignment_info['rotation_matrices'].append(rotation_matrix)
                
            elif alignment_method == 'procrustes':
                # Use Procrustes analysis for alignment
                aligned_AX_df, rotation_matrix = align_coordinate_systems_procrustes(AX_df, aw_df)
                aligned_aw_df = aw_df.copy()
                
                alignment_info['rotation_matrices'].append(rotation_matrix)
                
            elif alignment_method == 'none':
                # No alignment, use original data
                aligned_AX_df = AX_df.copy()
                aligned_aw_df = aw_df.copy()
                alignment_info['rotation_matrices'].append(np.eye(3))
            
            else:
                raise ValueError(f"Unknown alignment method: {alignment_method}")
            
            aligned_aw_data.append(aligned_aw_df)
            aligned_AX_data.append(aligned_AX_df)
            
            # Record sample statistics
            sample_stats = {
                'sample_id': sample['sample_id'],
                'filename_aw': sample['filename_aw'],
                'filename_AX': sample['filename_AX'],
                'length': len(aligned_aw_df),
                'aw_mean_accel': np.mean([aligned_aw_df['accelX'].mean(), 
                                        aligned_aw_df['accelY'].mean(), 
                                        aligned_aw_df['accelZ'].mean()]),
                'AX_mean_accel': np.mean([aligned_AX_df['accelX'].mean(), 
                                           aligned_AX_df['accelY'].mean(), 
                                           aligned_AX_df['accelZ'].mean()])
            }
            alignment_info['sample_stats'].append(sample_stats)
            
            logger.debug(f"Processed sample {i+1}/{len(training_samples)}: {sample['filename_aw']}")
            
        except Exception as e:
            logger.error(f"Error processing sample {i+1} ({sample.get('filename_aw', 'unknown')}): {str(e)}")
            continue
    
    logger.info(f"Successfully processed {len(aligned_aw_data)} samples with {alignment_method} alignment")
    return aligned_aw_data, aligned_AX_data, alignment_info


def calculate_rotation_matrix_from_sample(AX_df: pd.DataFrame, aw_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate rotation matrix from a single paired sample using mean acceleration vectors
    
    @param AX_df: AX data DataFrame
    @param aw_df: Apple Watch data DataFrame
    @return: 3x3 rotation matrix
    """
    AX_accel = AX_df[['accelX', 'accelY', 'accelZ']].values
    aw_accel = aw_df[['accelX', 'accelY', 'accelZ']].values
    
    # Calculate mean acceleration vectors (representing gravity direction)
    AX_vec = np.mean(AX_accel, axis=0)
    aw_vec = np.mean(aw_accel, axis=0)
    
    # Normalize vectors
    AX_vec = AX_vec / np.linalg.norm(AX_vec)
    aw_vec = aw_vec / np.linalg.norm(aw_vec)
    
    # Calculate rotation matrix using scipy
    rot_obj, rmsd = R_scipy.align_vectors([AX_vec], [aw_vec])
    rotation_matrix = rot_obj.as_matrix()
    
    return rotation_matrix


def train_mapping_model_from_paired_data(training_samples: list, alignment_method: str = 'rotation_matrix') -> tuple:
    """
    Train mapping model from paired training data with coordinate alignment
    
    @param training_samples: List of paired training samples
    @param alignment_method: Method for coordinate system alignment
    @return: Tuple of (trained_model, alignment_info, training_metrics)
    """
    logger.info("Starting mapping model training from paired data...")
    
    # Process samples with alignment
    aligned_aw_data, aligned_AX_data, alignment_info = process_paired_samples_with_alignment(
        training_samples, alignment_method
    )
    
    if not aligned_aw_data or not aligned_AX_data:
        raise ValueError("No successfully aligned data found for training")
    
    # Combine all aligned data for training
    all_AX_features = []
    all_aw_targets = []
    
    for AX_df, aw_df in zip(aligned_AX_data, aligned_aw_data):
        AX_features = AX_df[['accelX', 'accelY', 'accelZ']].values
        aw_targets = aw_df[['accelX', 'accelY', 'accelZ']].values
        
        all_AX_features.append(AX_features)
        all_aw_targets.append(aw_targets)
    
    # Stack all data
    X = np.vstack(all_AX_features)  # AX data as input
    y = np.vstack(all_aw_targets)      # AW data as target
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Train SVR models for each axis
    svr_models = []
    y_pred = np.zeros_like(y)
    for i in range(y.shape[1]):
        svr = SVR(kernel='rbf')
        svr.fit(X, y[:, i])
        svr_models.append(svr)
        y_pred[:, i] = svr.predict(X)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    training_metrics = {
        'rmse': rmse,
        'r2_score': r2,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_targets': y.shape[1]
    }
    
    logger.info(f"Mapping model training completed:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  Training samples: {X.shape[0]}")
    
    # 返回模型列表而不是单一模型
    return svr_models, alignment_info, training_metrics

def visualise(data, show=True, title="Acceleration Data"):
    plt.plot(data['timestamp'], data['accelX'], label='X-axis')
    plt.plot(data['timestamp'], data['accelY'], label='Y-axis')
    plt.plot(data['timestamp'], data['accelZ'], label='Z-axis')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid()
    if show:
        plt.show()

# ---------------------------------------
AW_SAMPLING_RATE = 20 # Hz, Apple Watch sampling rate
LAB_SAMPLING_RATE = 1500 # Hz, AX original sampling rate
CUTOFF_FREQ = 5 # Hz, Low-pass filter cutoff frequency
# ---------------------------------------

# --- 1. Data Preprocessing ---
def preprocess_acceleration_data(df: pd.DataFrame, original_sampling_rate: int, target_sampling_rate: int, cutoff_freq: float, filtered=False) -> pd.DataFrame:
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
    
    if not filtered:
        return resampled_df
    
    # 1.2 Highpass Filter
    nyquist_freq = 0.5 * target_sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(4, normal_cutoff, btype='highpass', analog=False) # 4th order Butterworth filter

    filtered_df = resampled_df.copy()
    for col in ['accelX', 'accelY', 'accelZ']:
        filtered_df[col] = filtfilt(b, a, resampled_df[col].values)
    return filtered_df
        

def normalize_data(df: pd.DataFrame, type=None) -> tuple:
    """
    Applies Z-score normalization to the 'accelX', 'accelY', and 'accelZ' columns.

    @param df: DataFrame with columns 'accelX', 'accelY', 'accelZ'
    @param type: Optional type for saving the scaler, e.g., 'aw' or 'lab'. If not specified, the scaler will not be saved.
    @return: Tuple of (normalized DataFrame, scaler used for normalization)
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[['accelX', 'accelY', 'accelZ']])
    normalized_df = df.copy()
    normalized_df[['accelX', 'accelY', 'accelZ']] = normalized_data
    if type:
        joblib.dump(scaler, rf'./mapping/cache/scaler_{type}.joblib') # Save the scaler for later use
    return normalized_df, scaler


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

def calculate_rotation_matrix_from_flat_pose(AX_flat_df, aw_flat_df):
    """
    根据平躺姿态数据计算从AX到Apple Watch的旋转矩阵。
    假设在平躺姿态下，两个设备都静止，且各自的Z轴都近似指向（或反向指向）重力方向。
    更通用的方法是，两个设备的Y轴（通常是设备的前向）都指向某个共同的固定方向（例如受试者头部）。
    
    这里我们使用重力向量作为参考：
    - AX的平均重力向量 (AX_vec)
    - Apple Watch的平均重力向量 (aw_vec)
    我们将计算一个旋转矩阵，使得 AX_vec 旋转后与 aw_vec 方向一致。
    
    更精确的校准会使用多于一个姿态，但平躺是最简单的单姿态校准。
    
    返回一个3x3的旋转矩阵 R，使得 `R @ AX_accel` 能够对齐到 aw_accel 的坐标系。
    """
    AX_accel_flat = AX_flat_df[['accelX', 'accelY', 'accelZ']].values
    aw_accel_flat = aw_flat_df[['accelX', 'accelY', 'accelZ']].values

    # 计算平均加速度向量 (代表重力方向在各自坐标系下的投影)
    AX_vec = np.mean(AX_accel_flat, axis=0)
    aw_vec = np.mean(aw_accel_flat, axis=0)

    # 归一化向量 (只关心方向)
    AX_vec = AX_vec / np.linalg.norm(AX_vec)
    aw_vec = aw_vec / np.linalg.norm(aw_vec)

    print()
    print("AX source vector:", AX_vec)
    print("Apple Watch target vector:", aw_vec)
    print()

    # 计算旋转矩阵（将 AX_vec 旋转到 aw_vec）
    # 使用 scipy.spatial.transform.Rotation.align_vectors
    # 它能找到将一组向量旋转到另一组向量的最佳旋转
    rot_obj, rmsd = R_scipy.align_vectors([AX_vec], [aw_vec])
    rotation_matrix = rot_obj.as_matrix()
    print(f"[Rotation Matrix] RMSD (Root Mean Square Deviation): {rmsd:.4f}")
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
    @param series1: 第一个时间序列
    @param series2: 第二个时间序列
    @return: DTW对齐结果，包含扭曲路径和距离。使用方式：alignment.index1 和 alignment.index2
    """
    alignment = dtw(x=series1, y=series2, 
                    dist_method="euclidean",
                    step_pattern="symmetric2")
    return alignment

# --- 4. 映射模型训练函数 ---
def train_mapping_model(aligned_AX_accel: pd.DataFrame, aligned_aw_accel: pd.DataFrame) -> LinearRegression:
    """
    训练一个线性回归模型，将AX加速度映射到Apple Watch加速度。
    输入为AX数据，输出为Apple Watch数据。
    """
    # 将输入数据转换为 (n_samples, n_features) 形状
    X = aligned_AX_accel[['accelX', 'accelY', 'accelZ']].values  # 输入是AX数据
    y = aligned_aw_accel[['accelX', 'accelY', 'accelZ']].values     # 目标是Apple Watch的XYZ

    # 训练多输出线性回归模型
    model = LinearRegression()
    model.fit(X, y)
    
    # 评估模型
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"Mapping Model Training Result (AX -> Apple Watch):")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    return model


if __name__ == "__main__":
    # ============================================================================
    # 新的训练流程：使用配对训练数据
    # ============================================================================
    
    print("=== 使用配对训练数据进行映射模型训练 ===")
    
    # --- 1. 加载配对训练数据 ---
    paired_data_path = "paired_training_data/paired_training_data.pkl"
    
    try:
        training_samples = load_paired_training_data(paired_data_path)
        print(f"成功加载 {len(training_samples)} 个配对训练样本")
        
        # 显示样本信息
        print("\n配对样本概览:")
        for i, sample in enumerate(training_samples[:5]):  # 显示前5个样本
            print(f"  样本 {sample['sample_id']:2d}: {sample['filename_aw']:25s} <-> {sample['filename_AX']:25s} "
                  f"({sample['length']:4d} 数据点, {sample['duration']:.1f}秒)")
        if len(training_samples) > 5:
            print(f"  ... 还有 {len(training_samples) - 5} 个样本")
    
    except FileNotFoundError:
        print(f"错误: 找不到配对训练数据文件 {paired_data_path}")
        print("请先运行 build_training_set.py 生成配对训练数据")
        exit(1)
    
    # --- 2. 选择对齐方法并训练模型 ---
    alignment_methods = ['rotation_matrix', 'procrustes', 'none']
    
    for method in alignment_methods:
        print(f"\n--- 使用 {method} 对齐方法训练模型 ---")
        
        try:
            # 训练模型
            model, alignment_info, metrics = train_mapping_model_from_paired_data(
                training_samples, alignment_method=method
            )
            
            # 保存模型和相关信息
            model_dir = f"mapping_models_{method}"
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存训练好的模型
            model_path = os.path.join(model_dir, 'mapping_model.joblib')
            joblib.dump(model, model_path)
            
            # 保存对齐信息
            alignment_path = os.path.join(model_dir, 'alignment_info.pkl')
            with open(alignment_path, 'wb') as f:
                pickle.dump(alignment_info, f)
            
            # 保存训练指标
            metrics_path = os.path.join(model_dir, 'training_metrics.pkl')
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            
            print(f"模型和相关文件已保存到 {model_dir}/ 目录")
            print(f"  - 模型文件: {model_path}")
            print(f"  - 对齐信息: {alignment_path}")
            print(f"  - 训练指标: {metrics_path}")
            
            # 显示训练结果
            print(f"\n{method} 方法训练结果:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R² Score: {metrics['r2_score']:.4f}")
            print(f"  训练样本数: {metrics['n_samples']}")
            
        except Exception as e:
            print(f"使用 {method} 方法训练时出错: {str(e)}")
            continue
    
    # --- 3. 模型比较和可视化 ---
    print(f"\n--- 模型性能比较 ---")
    
    model_results = {}
    
    for method in alignment_methods:
        metrics_path = f"mapping_models_{method}/training_metrics.pkl"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
            model_results[method] = metrics
    
    if model_results:
        print("方法对比:")
        print(f"{'方法':<15} {'RMSE':<10} {'R² Score':<10} {'样本数':<10}")
        print("-" * 50)
        
        best_method = None
        best_r2 = -float('inf')
        
        for method, metrics in model_results.items():
            print(f"{method:<15} {metrics['rmse']:<10.4f} {metrics['r2_score']:<10.4f} {metrics['n_samples']:<10}")
            if metrics['r2_score'] > best_r2:
                best_r2 = metrics['r2_score']
                best_method = method
        
        print(f"\n最佳方法: {best_method} (R² = {best_r2:.4f})")
        
        # 创建性能比较图
        methods = list(model_results.keys())
        rmse_values = [model_results[m]['rmse'] for m in methods]
        r2_values = [model_results[m]['r2_score'] for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE 比较
        ax1.bar(methods, rmse_values, color='skyblue', alpha=0.7)
        ax1.set_title('Model RMSE Comparison')
        ax1.set_ylabel('RMSE')
        ax1.set_xlabel('Alignment Method')
        for i, v in enumerate(rmse_values):
            ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
        
        # R² Score 比较
        ax2.bar(methods, r2_values, color='lightcoral', alpha=0.7)
        ax2.set_title('Model R² Score Comparison')
        ax2.set_ylabel('R² Score')
        ax2.set_xlabel('Alignment Method')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(r2_values):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"性能比较图已保存为 model_performance_comparison.png")
    
    # --- 4. 测试最佳模型 ---
    if best_method and len(training_samples) > 0:
        print(f"\n--- 测试最佳模型 ({best_method}) ---")
        
        # 加载最佳模型
        best_model_path = f"mapping_models_{best_method}/mapping_model.joblib"
        best_model = joblib.load(best_model_path)
        
        # 使用第一个训练样本进行测试演示
        test_sample = training_samples[0]
        aw_df, AX_df = convert_sample_to_dataframes(test_sample)
        
        # 应用相同的对齐方法
        if best_method == 'rotation_matrix':
            rotation_matrix = calculate_rotation_matrix_from_sample(AX_df, aw_df)
            aligned_AX_df = apply_rotation_matrix(AX_df, rotation_matrix)
        elif best_method == 'procrustes':
            aligned_AX_df, _ = align_coordinate_systems_procrustes(AX_df, aw_df)
        else:
            aligned_AX_df = AX_df.copy()
        
        # 进行预测
        AX_input = aligned_AX_df[['accelX', 'accelY', 'accelZ']].values
        
        # 检查模型类型并进行相应的预测
        if isinstance(best_model, list):
            # SVR 模型列表，需要为每个轴分别预测
            aw_pred = np.zeros((AX_input.shape[0], 3))
            for i, svr_model in enumerate(best_model):
                aw_pred[:, i] = svr_model.predict(AX_input)
        else:
            # 单一模型（如 LinearRegression）
            aw_pred = best_model.predict(AX_input)
        
        aw_true = aw_df[['accelX', 'accelY', 'accelZ']].values
        
        # 计算测试误差
        test_rmse = np.sqrt(mean_squared_error(aw_true, aw_pred))
        test_r2 = r2_score(aw_true, aw_pred)
        
        print(f"测试样本: {test_sample['filename_aw']}")
        print(f"测试 RMSE: {test_rmse:.4f}")
        print(f"测试 R² Score: {test_r2:.4f}")
        
        # 可视化预测结果
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        axis_labels = ['X', 'Y', 'Z', 'Magnitude']
        
        for i in range(3):
            axes[i].plot(aw_true[:, i], label=f'Real AW {axis_labels[i]}', alpha=0.8)
            axes[i].plot(aw_pred[:, i], label=f'Converted AW {axis_labels[i]}', linestyle='--', alpha=0.8)
            axes[i].set_title(f'{axis_labels[i]} Axis Acceleration Prediction Comparison')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Acceleration')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # 合成加速度对比
        true_mag = np.linalg.norm(aw_true, axis=1)
        pred_mag = np.linalg.norm(aw_pred, axis=1)
        axes[3].plot(true_mag, label='Real Combined Acceleration', alpha=0.8)
        axes[3].plot(pred_mag, label='Converted Combined Acceleration', linestyle='--', alpha=0.8)
        axes[3].set_title('Combined Acceleration Prediction Comparison')
        axes[3].set_xlabel('Time Step')
        axes[3].set_ylabel('Acceleration')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'prediction_test_{best_method}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"预测结果可视化已保存为 prediction_test_{best_method}.png")
    
    print("\n=== 配对数据训练流程完成 ===")
    
    
    # ============================================================================
    # 原有的训练流程（注释掉，但保留作为参考）
    # ============================================================================
    exit()  # 退出，不执行原有流程
    
    r"""     
    print("--- 0. 加载平躺姿态数据 ---")
    # root_dir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/lying_data/'
    root_dir = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\lying_data\\"
    flat_AX_filepath = root_dir + 'L05 Lying01.c3d.ontrackclassifier.joblib'
    AX_flat_data_raw = load_lab_file(flat_AX_filepath, side='right')
    flat_aw_filepath = root_dir + 'GERF-L-D524-M3-S0041.csv' # self collected Apple Watch data, with the same lying pose as AX.
    aw_flat_data_raw = load_aw_file(flat_aw_filepath)

    # 可视化平躺姿态数据，并根据可视化结果粗略截取apple watch数据，使得两者的时间长度大体一致
    # visualise(AX_flat_data_raw, title="AX Flat Pose Data")
    # visualise(aw_flat_data_raw, title="Apple Watch Flat Pose Data")
    # exit()

    # 手动截取
    AX_flat_data_raw = AX_flat_data_raw[(AX_flat_data_raw['timestamp'] >= 2) & (AX_flat_data_raw['timestamp'] <= 39)]
    AX_flat_data_raw['timestamp'] -= AX_flat_data_raw['timestamp'].min() # 将AX的时间戳转换为相对时间（从0开始）
    # 将aw的所有加速度从g转化为m/s^2
    aw_flat_data_raw['accelX'] *= 9.81
    aw_flat_data_raw['accelY'] *= 9.81
    aw_flat_data_raw['accelZ'] *= 9.81
    aw_flat_data_raw = aw_flat_data_raw[(aw_flat_data_raw['timestamp'] >= 122) & (aw_flat_data_raw['timestamp'] <= 159)]
    aw_flat_data_raw['timestamp'] -= aw_flat_data_raw['timestamp'].min() # 将aw的时间戳转换为相对时间（从0开始）

    # 对平躺姿态数据进行预处理（降采样到目标采样率）；因为load data的时候已经过滤过了所以filtered=False
    AX_flat_processed = preprocess_acceleration_data(
        AX_flat_data_raw, LAB_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ, filtered=False
    )
    aw_flat_processed = preprocess_acceleration_data(
        aw_flat_data_raw, AW_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ, filtered=False
    )

    # plt.subplot(2, 1, 1)
    # visualise(AX_flat_processed, show=False, title="AX Flat Pose Processed Data")
    # plt.subplot(2, 1, 2)
    # plt.tight_layout()
    # visualise(aw_flat_processed, title="Apple Watch Flat Pose Processed Data")


    print("\n--- 1. 计算坐标系旋转矩阵 ---")
    # AX 是源设备，Apple Watch 是目标设备
    rotation_matrix_AX_to_aw = calculate_rotation_matrix_from_flat_pose(
        AX_flat_processed, aw_flat_processed
    )
    print("旋转矩阵计算完成 (AX -> Apple Watch):", rotation_matrix_AX_to_aw)

    exit()
    """
    
    # --- 1. 定义坐标系旋转矩阵 ---
    # 这里的旋转矩阵已经由上面的代码基于平躺姿态数据计算得到
    rotation_matrix_AX_to_aw = np.array([[ 0.58377147,  0.28237329,  0.76123334],
                                            [ 0.36897715,  0.74289845, -0.55853179],
                                            [-0.72323352,  0.60693263,  0.32949362]])

    
    # --- 2. 加载运动数据 ---
    # calib_data_dir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/Calibration Data/'
    calib_data_dir = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\Calibration Data\\"
    motion_AX_filepath = os.path.join(calib_data_dir, 'Lab/LabChopped/chopped_right_M2TestingDrinking03.csv')
    motion_aw_filepath = os.path.join(calib_data_dir, 'AppleWatch/AW_Chopped/chopped_M2-S0079.csv')
    AX_motion_data_raw = load_csv_data(motion_AX_filepath, data_type='lab')
    aw_motion_data_raw = load_aw_file(motion_aw_filepath)

    print(f"AX 运动数据点数: {len(AX_motion_data_raw)}")
    print(f"Apple Watch 运动数据点数: {len(aw_motion_data_raw)}")
    
    # --- 3. 预处理运动数据 ---
    AX_motion_processed = preprocess_acceleration_data(
        AX_motion_data_raw, LAB_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ
    )
    aw_motion_processed = preprocess_acceleration_data(
        aw_motion_data_raw, AW_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ
    )
    
    # 对预处理后的数据进行归一化 (可选，但在某些情况下对DTW和映射有帮助)
    AX_motion_normalized, AX_scaler = normalize_data(AX_motion_processed, type='AX')
    aw_motion_normalized, aw_scaler = normalize_data(aw_motion_processed, type='aw')

    print(f"AX 归一化后数据点数 ({AW_SAMPLING_RATE}Hz): {len(AX_motion_normalized)}")
    print(f"Apple Watch 归一化后数据点数 ({AW_SAMPLING_RATE}Hz): {len(aw_motion_normalized)}")

    # --- 4. 应用坐标系旋转 ---
    # 将 AX 运动数据旋转到 Apple Watch 的坐标系
    AX_motion_rotated = apply_rotation_matrix(AX_motion_normalized, rotation_matrix_AX_to_aw)
    aw_motion_rotated = aw_motion_normalized.copy()  # Apple Watch数据不需要旋转
    
    # 如果需要，进行左右手镜像处理 (例如，AX在右手，AW在左手)
    # AX_motion_rotated = mirror_data_for_hand(AX_motion_rotated, hand_type='right')


    # --- 5. 初始时间对齐 (粗略对齐) ---
    # 手动进行粗略对齐！
    valid_start, valid_end = 0, len(AX_motion_rotated) # 暂且使用整个数据段
    AX_accel_for_dtw = AX_motion_rotated[['accelX', 'accelY', 'accelZ']].values[valid_start:valid_end]
    valid_start, valid_end = 0, len(aw_motion_rotated)
    aw_accel_for_dtw = aw_motion_rotated[['accelX', 'accelY', 'accelZ']].values[valid_start:valid_end]


    # --- 6. 精细时间对齐 (DTW) ---
    print("\n--- 精细时间对齐 (DTW) ---")
    # DTW 对齐坐标系后的数据
    alignment_result = align_with_dtw(AX_accel_for_dtw, aw_accel_for_dtw)
    
    print(f"DTW 距离: {alignment_result.distance:.4f}")

    # 获取对齐后的序列（通过 DTW 路径重新采样）
    aligned_AX_series = AX_accel_for_dtw[alignment_result.index1]
    aligned_aw_series = aw_accel_for_dtw[alignment_result.index2]

    print(f"DTW 对齐后 AX 序列长度: {len(aligned_AX_series)}")
    print(f"DTW 对齐后 Apple Watch 序列长度: {len(aligned_aw_series)}")

    # 可视化对齐效果
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axis_labels = ['X', 'Y', 'Z']
    for i, ax in enumerate(axs.flat[:3]):
        ax.plot(aligned_AX_series[:, i], label=f'AX {axis_labels[i]}-axis (DTW Aligned, Rotated)')
        ax.plot(aligned_aw_series[:, i], label=f'Apple Watch {axis_labels[i]}-axis (DTW Aligned)', linestyle='--')
        ax.set_title(f"{axis_labels[i]}-axis Acceleration after DTW Alignment")
        ax.set_xlabel("Aligned Sample Index")
        ax.set_ylabel("Acceleration (normalized)")
        ax.legend()
        ax.grid(True)
    # 合成加速度
    AX_magnitude = np.linalg.norm(aligned_AX_series, axis=1)
    aw_magnitude = np.linalg.norm(aligned_aw_series, axis=1)
    axs[1, 1].plot(AX_magnitude, label='AX Magnitude (DTW Aligned, Rotated)')
    axs[1, 1].plot(aw_magnitude, label='Apple Watch Magnitude (DTW Aligned)', linestyle='--')
    axs[1, 1].set_title("Magnitude Acceleration after DTW Alignment")
    axs[1, 1].set_xlabel("Aligned Sample Index")
    axs[1, 1].set_ylabel("Acceleration (normalized)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    plt.tight_layout()
    plt.show()

    # --- 7. 映射模型训练 ---
    print("\n--- 7. 映射模型训练 ---")
    # 使用DTW对齐后的数据训练映射模型
    mapping_model = train_mapping_model(pd.DataFrame(aligned_AX_series, columns=['accelX','accelY','accelZ']), 
                                        pd.DataFrame(aligned_aw_series, columns=['accelX','accelY','accelZ']))
    
    # 保存模型
    model_save_path = './mapping/cache/mapping_model_AX_to_aw.joblib'
    joblib.dump(mapping_model, model_save_path)
    print(f"映射模型已保存到: {model_save_path} (AX -> Apple Watch)")
    
    exit()
    

    # --- 8. 映射模型应用示例 ---
    print("\n--- 8. 映射模型应用示例 ---")
    # 因为AX是高精度设备，我们假设它是“源”或“输入”，Apple Watch是“目标”或“输出”
    # 加载mapping_model
    mapping_model = joblib.load('mapping_model.joblib')
    
    # 从 aligned_AX_series 中取一些点进行预测演示
    sample_AX_input = load_calib_lab('/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/Calibration Data/Lab/LabChopped/chopped_right_M2TestingReading02.csv')
    sample_AX_input = preprocess_acceleration_data(sample_AX_input, LAB_SAMPLING_RATE, AW_SAMPLING_RATE, CUTOFF_FREQ)
    # normalize the sample input using the scaler
    scaler = joblib.load('scaler.joblib')
    sample_AX_input = scaler.transform(sample_AX_input[['accelX', 'accelY', 'accelZ']])
    sample_AX_input = pd.DataFrame(sample_AX_input, columns=['accelX', 'accelY', 'accelZ'])
    predicted_aw_accel = mapping_model.predict(sample_AX_input)

    print(f"AX 输入:\n{sample_AX_input}")
    print(f"预测的 Apple Watch 加速度:\n{predicted_aw_accel}")
    
    # 与真实的 Apple Watch 对齐数据进行比较
    true_aw_accel = load_aw_file('/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/Calibration Data/AppleWatch/AW_Chopped/chopped_M3-S0078.csv')
    print(f"真实的 Apple Watch 加速度:\n{true_aw_accel}")

    # 评估预测准确性
    mse_sample = mean_squared_error(true_aw_accel, predicted_aw_accel)
    print(f"示例预测的均方误差 (MSE): {mse_sample:.4f}")

    # print("\n--- 流程完成 ---")