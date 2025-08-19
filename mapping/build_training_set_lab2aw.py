""" Concatenate the data from aw and vicon to build a larger training set than a single sample. """

import os
import sys
sys.path.insert(0, r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline")
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
import traceback
logger = getLogger('INFO')

from config import AW_SAMPLING_RATE, LAB_SAMPLING_RATE


def load_csv_data(csv_path: str, data_type: str) -> pd.DataFrame:
    """
    Load calibration data from a CSV file.
    
    @param csv_path: Path to the CSV file containing calibration data.
    @param data_type: Type of data, either 'aw' for Apple Watch or 'lab' for Vicon Lab.
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


def get_csv_files(directory: str) -> list:
    """
    Get all CSV files from a directory.
    
    @param directory: Path to the directory to search for CSV files.
    @return: List of CSV file paths.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return csv_files


def match_files_by_name(aw_files: list, vicon_files: list, matching_dict: dict) -> list:
    """
    Match AW and Vicon files based on filename patterns and a matching dictionary.
    
    @param aw_files: List of AW CSV file paths.
    @param vicon_files: List of Vicon CSV file paths.
    @param matching_dict: Dictionary mapping patterns like "M1-S0077": "Walking01".
    @return: List of tuples (aw_file, vicon_file) in matching order.
    """
    matched_pairs = []
    
    for key, value in matching_dict.items():
        aw_file = None
        vicon_file = None
        
        # Find matching AW file
        for file in aw_files:
            filename = os.path.basename(file)
            if key in filename or value in filename:
                aw_file = file
                break
        
        # Find matching Vicon file
        for file in vicon_files:
            filename = os.path.basename(file)
            if key in filename or value in filename:
                vicon_file = file
                break
        
        if aw_file and vicon_file:
            matched_pairs.append((aw_file, vicon_file))
        else:
            logger.warning(f"No matching pair found for key: {key}, value: {value}")
    
    return matched_pairs


def sync_and_resample_data(aw_data: pd.DataFrame, vicon_data: pd.DataFrame) -> tuple:
    """
    Synchronize and resample AW and Vicon data to the same time base and length.

    @param aw_data: Apple Watch data DataFrame.
    @param vicon_data: Vicon data DataFrame.
    @return: Tuple of (synchronized_aw_df, synchronized_vicon_df).
    """
    assert len(aw_data) <= len(vicon_data), "AW data must be shorter or equal to Vicon data."

    target_length = len(aw_data)

    # 重采样 Vicon 数据
    vicon_resampled = {}
    for col in ['accelX', 'accelY', 'accelZ']:
        vicon_resampled[col] = resample(vicon_data[col], target_length)
    # 时间戳也用 AW 的
    vicon_resampled['timestamp'] = aw_data['timestamp'].values

    vicon_sync = pd.DataFrame(vicon_resampled, columns=['timestamp', 'accelX', 'accelY', 'accelZ'])
    aw_sync = aw_data.reset_index(drop=True)

    return aw_sync, vicon_sync


def create_paired_training_samples(file_pairs: list) -> list:
    """
    Create paired training samples from matched file pairs with synchronized data.
    
    @param file_pairs: List of tuples (aw_file, vicon_file).
    @return: List of dictionaries containing paired training samples.
    """
    training_samples = []
    
    for i, (aw_file, vicon_file) in enumerate(file_pairs):
        try:
            # Load data
            aw_data = load_csv_data(aw_file, 'aw')
            vicon_data = load_csv_data(vicon_file, 'lab')
            
            logger.info(f"Processing pair {i+1}: {os.path.basename(aw_file)} ({len(aw_data)} rows) <-> {os.path.basename(vicon_file)} ({len(vicon_data)} rows)")
            
            # Synchronize data
            aw_sync, vicon_sync = sync_and_resample_data(aw_data, vicon_data)
            
            if aw_sync.empty or vicon_sync.empty:
                logger.warning(f"Skipping pair {i+1} due to synchronization failure")
                continue
            
            # Create paired sample
            sample = {
                'sample_id': i,
                'filename_aw': os.path.basename(aw_file),
                'filename_vicon': os.path.basename(vicon_file),
                'length': len(aw_sync),
                'duration': aw_sync['timestamp'].max() - aw_sync['timestamp'].min(),
                'timestamp': aw_sync['timestamp'].values,
                'aw_accelX': aw_sync['accelX'].values,
                'aw_accelY': aw_sync['accelY'].values,
                'aw_accelZ': aw_sync['accelZ'].values,
                'vicon_accelX': vicon_sync['accelX'].values,
                'vicon_accelY': vicon_sync['accelY'].values,
                'vicon_accelZ': vicon_sync['accelZ'].values
            }
            
            training_samples.append(sample)
            logger.info(f"Created paired sample {i+1}: {len(aw_sync)} synchronized data points")
            
        except Exception as e:
            logger.error(f"Error processing pair {i+1} ({os.path.basename(aw_file)}, {os.path.basename(vicon_file)}): {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    logger.success(f"Successfully created {len(training_samples)} paired training samples")
    return training_samples


def save_paired_training_data(training_samples: list, output_dir: str = ".") -> None:
    """
    Save paired training data in multiple formats.
    
    @param training_samples: List of paired training samples.
    @param output_dir: Output directory for saving files.
    """
    if not training_samples:
        logger.warning("No training samples to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Save as pickle file (recommended for complex data structures)
    import pickle
    pickle_path = os.path.join(output_dir, 'paired_training_data.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(training_samples, f)
    logger.info(f"Saved paired training data as pickle: {pickle_path}")
    
    # Method 2: Save as separate CSV files for each sample
    csv_dir = os.path.join(output_dir, 'paired_csv_samples')
    os.makedirs(csv_dir, exist_ok=True)
    
    for sample in training_samples:
        sample_id = sample['sample_id']

        # Create DataFrame for this sample
        sample_df = pd.DataFrame({
            'Timestamp': sample['timestamp'],
            'AW_accelX': sample['aw_accelX'],
            'AW_accelY': sample['aw_accelY'],
            'AW_accelZ': sample['aw_accelZ'],
            'Vicon_accelX': sample['vicon_accelX'],
            'Vicon_accelY': sample['vicon_accelY'],
            'Vicon_accelZ': sample['vicon_accelZ']
        })
        
        csv_path = os.path.join(csv_dir, f'sample_{sample_id:03d}_{sample["filename_aw"].replace(".csv", "")}.csv')
        sample_df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved {len(training_samples)} individual CSV files in: {csv_dir}")
    
    # Method 3: Save summary information
    summary_data = []
    for sample in training_samples:
        summary_data.append({
            'Sample_ID': sample['sample_id'],
            'AW_Filename': sample['filename_aw'],
            'Vicon_Filename': sample['filename_vicon'],
            'Length': sample['length'],
            'Duration_seconds': sample['duration']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'training_samples_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved training samples summary: {summary_path}")


def build_training_set(aw_directory: str, vicon_directory: str, matching_dict: dict, 
                      output_dir: str = "paired_training_data"):
    """
    Build paired training set from matched AW and Vicon data files.
    
    @param aw_directory: Directory containing AW CSV files.
    @param vicon_directory: Directory containing Vicon CSV files.
    @param matching_dict: Dictionary mapping patterns for file matching.
    @param output_dir: Output directory for saving paired training data.
    """
    logger.info("Starting to build paired training set...")
    
    # Get all CSV files from both directories
    aw_files = get_csv_files(aw_directory)
    vicon_files = get_csv_files(vicon_directory)
    
    logger.info(f"Found {len(aw_files)} AW files and {len(vicon_files)} Vicon files")
    
    # Match files based on the dictionary
    matched_pairs = match_files_by_name(aw_files, vicon_files, matching_dict)
    
    if not matched_pairs:
        logger.error("No matched file pairs found!")
        return
    
    logger.success(f"Successfully matched {len(matched_pairs)} file pairs")
    
    # Create paired training samples
    training_samples = create_paired_training_samples(matched_pairs)
    
    if not training_samples:
        logger.error("No training samples were created!")
        return
    
    # Save paired training data
    save_paired_training_data(training_samples, output_dir)
    
    logger.info("Paired training set building completed!")
    
    return training_samples

    return training_samples


if __name__ == "__main__":
    # Example usage
    # Define your matching dictionary
    matching_dict = {
        "M1-S0077": "Walking01",
        "M2-S0077": "Drinking01",
        "M3-S0077": "Reading01",
        "M1-S0078": "Walking02",
        "M2-S0078": "Drinking02",
        "M3-S0078": "Reading02",
        # "M1-S0079": "Walking03",
        "M2-S0079": "Drinking03",
        "M3-S0079": "Reading03",
        # "M2-S0080": "Drinking03",
    }
    
    # Define directories
    aw_directory = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\Calibration Data\AppleWatch\AW_Chopped"
    vicon_directory = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\Calibration Data\Lab\LabChopped"
    
    # Build paired training set
    training_samples = build_training_set(
        aw_directory=aw_directory,
        vicon_directory=vicon_directory,
        matching_dict=matching_dict,
        output_dir="paired_training_data"
    )
    
    # Optional: Print summary of created samples
    if training_samples:
        print(f"\n=== Training Set Summary ===")
        print(f"Total paired samples: {len(training_samples)}")
        total_duration = sum(sample['duration'] for sample in training_samples)
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Average sample length: {np.mean([sample['length'] for sample in training_samples]):.1f} data points")
        print("\nSample details:")
        for sample in training_samples:
            print(f"  Sample {sample['sample_id']:2d}: {sample['filename_aw']} <-> {sample['filename_vicon']:33s} ({sample['length']} pts, {sample['duration']:.1f}s)")

