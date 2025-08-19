import os
import sys
sys.path.insert(0, r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline")
import pandas as pd
import numpy as np
from loguru import logger
from dataTransformer.filter import filter
from scipy.signal import resample
from config import getLogger
import traceback
logger = getLogger('INFO')

from config import AW_SAMPLING_RATE, AX_SAMPLING_RATE


def load_csv_data(csv_path: str, data_type: str) -> pd.DataFrame:
    """
    Load calibration data from a CSV file.
    
    @param csv_path: Path to the CSV file containing calibration data.
    @param data_type: Type of data, either 'aw' for Apple Watch or 'AX' for AXivity.
    @return: DataFrame with columns 'accelX', 'accelY', 'accelZ', and 'accel'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Calibration CSV file not found: {csv_path}")
    if data_type not in ['aw', 'AX']:
        raise ValueError("data_type must be either 'aw' or 'AX'.")
    
    data = pd.read_csv(csv_path)

    # determine timestamp
    timestamp_col = None
    for col in data.columns:
        if 'timestamp' in col.lower() or 'time' in col.lower():
            timestamp_col = col
            break
    if not timestamp_col:
        logger.warning(f"No timestamp column found in {os.path.basename(csv_path)}. Assuming sequential timestamps.")
        sampling_rate = AW_SAMPLING_RATE if data_type.lower()=='aw' else AX_SAMPLING_RATE
        data['Timestamp'] = pd.Series(range(len(data))) / sampling_rate
        timestamp_col = 'Timestamp'
    # determine acceleration for each AXis
    for col in data.columns:
        col_lower = col.lower()
        if '1' in col_lower or 'accelerationx' in col_lower:
            accel_x_col = col
        elif '2' in col_lower or 'accelerationy' in col_lower:
            accel_y_col = col
        elif '3' in col_lower or 'accelerationz' in col_lower:
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


def match_files_by_name(aw_files: list, AX_left_files: list, AX_right_files: list, matching_dict: dict) -> list:
    """
    Match AW and AX files based on filename patterns and a matching dictionary.
    Files with S0077 or S0078 are matched with left AX files, S0079 or S0080 with right AX files.
    
    @param aw_files: List of AW CSV file paths.
    @param AX_left_files: List of AX CSV file paths from left directory.
    @param AX_right_files: List of AX CSV file paths from right directory.
    @param matching_dict: Dictionary mapping patterns like "M1-S0077": "Walking01".
    @return: List of tuples (aw_file, AX_file) in matching order.
    """
    matched_pairs = []
    
    for key, value in matching_dict.items():
        aw_file = None
        AX_file = None

        # Find matching AW file
        for file in aw_files:
            filename = os.path.basename(file)
            if key in filename or value in filename:
                aw_file = file
                break

        if aw_file:
            # Determine which AX directory to use based on subject ID in the key
            if "S0077" in key or "S0078" in key:
                # Left hand - use left AX files
                AX_files_to_search = AX_left_files
                logger.info(f"Matching {key} with left AX files (left hand)")
            elif "S0079" in key or "S0080" in key:
                # Right hand - use right AX files
                AX_files_to_search = AX_right_files
                logger.info(f"Matching {key} with right AX files (right hand)")
            else:
                logger.warning(f"Subject ID in {key} not recognized, skipping...")
                continue

            # Find matching AX file from the appropriate directory
            for file in AX_files_to_search:
                filename = os.path.basename(file)
                if key in filename or value in filename:
                    AX_file = file
                    break
        
        if aw_file and AX_file:
            matched_pairs.append((aw_file, AX_file))
            logger.info(f"Matched pair: {os.path.basename(aw_file)} <-> {os.path.basename(AX_file)}")
        else:
            logger.warning(f"No matching pair found for key: {key}, value: {value}")
    
    return matched_pairs


def sync_and_resample_data(aw_data: pd.DataFrame, AX_data: pd.DataFrame) -> tuple:
    """
    Synchronize and resample AW and AX data to the same time base and length.

    @param aw_data: Apple Watch data DataFrame.
    @param AX_data: AX data DataFrame.
    @return: Tuple of (synchronized_aw_df, synchronized_AX_df).
    """
    assert len(aw_data) <= len(AX_data), "AW data must be shorter or equal to AX data."

    target_length = len(aw_data)

    # 重采样 AX 数据
    AX_resampled = {}
    for col in ['accelX', 'accelY', 'accelZ']:
        AX_resampled[col] = resample(AX_data[col], target_length)
    # 时间戳也用 AW 的
    AX_resampled['timestamp'] = aw_data['timestamp'].values

    AX_sync = pd.DataFrame(AX_resampled, columns=['timestamp', 'accelX', 'accelY', 'accelZ'])
    aw_sync = aw_data.reset_index(drop=True)

    return aw_sync, AX_sync


def create_paired_training_samples(file_pairs: list) -> list:
    """
    Create paired training samples from matched file pairs with synchronized data.
    
    @param file_pairs: List of tuples (aw_file, AX_file).
    @return: List of dictionaries containing paired training samples.
    """
    training_samples = []

    for i, (aw_file, AX_file) in enumerate(file_pairs):
        try:
            # Load data
            aw_data = load_csv_data(aw_file, 'aw')
            AX_data = load_csv_data(AX_file, 'AX')

            logger.info(f"Processing pair {i+1}: {os.path.basename(aw_file)} ({len(aw_data)} rows) <-> {os.path.basename(AX_file)} ({len(AX_data)} rows)")

            # Synchronize data
            aw_sync, AX_sync = sync_and_resample_data(aw_data, AX_data)

            if aw_sync.empty or AX_sync.empty:
                logger.warning(f"Skipping pair {i+1} due to synchronization failure")
                continue
            
            # Create paired sample
            sample = {
                'sample_id': i,
                'filename_aw': os.path.basename(aw_file),
                'filename_AX': os.path.basename(AX_file),
                'length': len(aw_sync),
                'duration': aw_sync['timestamp'].max() - aw_sync['timestamp'].min(),
                'timestamp': aw_sync['timestamp'].values,
                'aw_accelX': aw_sync['accelX'].values,
                'aw_accelY': aw_sync['accelY'].values,
                'aw_accelZ': aw_sync['accelZ'].values,
                'AX_accelX': AX_sync['accelX'].values,
                'AX_accelY': AX_sync['accelY'].values,
                'AX_accelZ': AX_sync['accelZ'].values
            }
            
            training_samples.append(sample)
            logger.info(f"Created paired sample {i+1}: {len(aw_sync)} synchronized data points")
            
        except Exception as e:
            logger.error(f"Error processing pair {i+1} ({os.path.basename(aw_file)}, {os.path.basename(AX_file)}): {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    logger.success(f"Successfully created {len(training_samples)} paired training samples")
    return training_samples


def save_paired_training_data(training_samples: list, output_dir: str) -> None:
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
    pickle_path = os.path.join(output_dir, 'paired_training_data_AX2aw.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(training_samples, f)
    logger.info(f"Saved paired training data as pickle: {pickle_path}")
    
    # Method 2: Save as separate CSV files for each sample
    csv_dir = os.path.join(output_dir, 'paired_csv_samples_AX2aw')
    os.makedirs(csv_dir, exist_ok=True)
    
    for sample in training_samples:
        sample_id = sample['sample_id']

        # Create DataFrame for this sample
        sample_df = pd.DataFrame({
            'Timestamp': sample['timestamp'],
            'AW_accelX': sample['aw_accelX'],
            'AW_accelY': sample['aw_accelY'],
            'AW_accelZ': sample['aw_accelZ'],
            'AX_accelX': sample['AX_accelX'],
            'AX_accelY': sample['AX_accelY'],
            'AX_accelZ': sample['AX_accelZ']
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
            'AX_Filename': sample['filename_AX'],
            'Length': sample['length'],
            'Duration_seconds': sample['duration']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'training_samples_summary_AX2aw.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved training samples summary: {summary_path}")


def build_training_set(aw_directory: str, AX_left_directory: str, AX_right_directory: str, matching_dict: dict, 
                      output_dir: str = "paired_training_data"):
    """
    Build paired training set from matched AW and AX data files.
    
    @param aw_directory: Directory containing AW CSV files.
    @param AX_left_directory: Directory containing left AX CSV files.
    @param AX_right_directory: Directory containing right AX CSV files.
    @param matching_dict: Dictionary mapping patterns for file matching.
    @param output_dir: Output directory for saving paired training data.
    """
    logger.info("Starting to build paired training set...")
    
    # Get all CSV files from both directories
    aw_files = get_csv_files(aw_directory)
    AX_left_files = get_csv_files(AX_left_directory)
    AX_right_files = get_csv_files(AX_right_directory)

    logger.info(f"Found {len(aw_files)} AW files, {len(AX_left_files)} left AX files, and {len(AX_right_files)} right AX files")

    # Match files based on the dictionary and hand position
    matched_pairs = match_files_by_name(aw_files, AX_left_files, AX_right_files, matching_dict)

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
    save_paired_training_data(training_samples, output_dir=output_dir)
    
    logger.info("Paired training set building completed!")
    
    return training_samples


if __name__ == "__main__":
    OUTPUT_DIR = r'E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline\mapping\paired_training_data'
    # Example usage
    # Define your matching dictionary
    matching_dict = {
        "M1-S0077": "Walking01",
        "M2-S0077": "Drinking01",
        "M3-S0077": "Reading01",
        "M1-S0078": "Walking02",
        "M2-S0078": "Drinking02",
        "M3-S0078": "Reading02",
        "M1-S0079": "Walking03",
        "M2-S0079": "Drinking03",
        "M3-S0079": "Reading03",
        "M2-S0080": "Drinking03",
    }
    
    # Define directories
    aw_directory = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\Calibration Data\AppleWatch\AW_Chopped"
    AX_right_directory = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\Calibration Data\Axivity\Right\106092_chopped"
    AX_left_directory = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\Calibration Data\Axivity\Left\93792_chopped"
    
    # Build paired training set
    training_samples = build_training_set(
        aw_directory=aw_directory,
        AX_left_directory=AX_left_directory,
        AX_right_directory=AX_right_directory,
        matching_dict=matching_dict,
        output_dir=OUTPUT_DIR
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
            print(f"  Sample {sample['sample_id']:2d}: {sample['filename_aw']} <-> {sample['filename_AX']:33s} ({sample['length']} pts, {sample['duration']:.1f}s)")

