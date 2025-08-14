import sys
sys.path.insert(0, r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline")

import os
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt

from config import getLogger
logger = getLogger('INFO')

LAB_SAMPLING_RATE = 1500

def low_pass_filter(data, cutoff_freq=5, sampling_rate=LAB_SAMPLING_RATE, order=4):
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
        logger.debug(f"Using {side} leg data from lab file")
    else:
        logger.warning("Lab file does not contain left/right data, using the first one in the structure.")
        data = list(lab_data.values())[0] # 选第一个
    
    # Extract acceleration data
    accel_x = data.get('accelX', [])
    accel_y = data.get('accelY', [])
    accel_z = data.get('accelZ', [])

    assert len(accel_x) == len(accel_y) == len(accel_z), "Acceleration data arrays must have the same length."
    assert len(accel_x) > 0, "Acceleration data cannot be empty."

    logger.debug(f"Applying low-pass filter to lab data with {len(accel_x)} samples.")
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

def analyze_acceleration_statistics(lab_data_dir: str, side: str = 'right') -> dict:
    """
    Analyze acceleration statistics across all lab data files in a directory.
    
    Parameters:
    - lab_data_dir: Directory containing lab data files (.joblib)
    - side: 'left' or 'right', specifies which leg's data to analyze
    
    Returns:
    - Dictionary containing acceleration statistics (mean, std, percentiles, etc.)
    """
    logger.info(f"Analyzing acceleration statistics in directory: {lab_data_dir}")
    
    if not os.path.exists(lab_data_dir):
        raise FileNotFoundError(f"Directory not found: {lab_data_dir}")
    
    all_accelerations = []
    all_std_values = []
    all_mean_values = []
    processed_files = 0
    
    # Process all .joblib files in the directory
    for filename in os.listdir(lab_data_dir):
        if filename.endswith('.joblib') and 'DESKTOP' not in filename:
            file_path = os.path.join(lab_data_dir, filename)
            try:
                logger.debug(f"Processing file: {filename}")
                df = load_lab_file(file_path, side=side)
                
                # Get acceleration magnitude
                accel_magnitude = df['accel'].values
                
                # Collect statistics
                all_accelerations.extend(accel_magnitude)
                all_std_values.append(np.std(accel_magnitude))
                all_mean_values.append(np.mean(accel_magnitude))
                processed_files += 1
                
            except Exception as e:
                logger.warning(f"Failed to process file {filename}: {str(e)}")
                continue
    
    if processed_files == 0:
        raise ValueError("No valid lab data files found in the directory")
    
    all_accelerations = np.array(all_accelerations)
    
    # Calculate comprehensive statistics
    stats = {
        'processed_files': processed_files,
        'total_samples': len(all_accelerations),
        'global_mean': np.mean(all_accelerations),
        'global_std': np.std(all_accelerations),
        'global_min': np.min(all_accelerations),
        'global_max': np.max(all_accelerations),
        'percentile_25': np.percentile(all_accelerations, 25),
        'percentile_33': np.percentile(all_accelerations, 33),
        'percentile_50': np.percentile(all_accelerations, 50),  # median
        'percentile_66': np.percentile(all_accelerations, 66),
        'percentile_75': np.percentile(all_accelerations, 75),
        'percentile_90': np.percentile(all_accelerations, 90),
        'percentile_95': np.percentile(all_accelerations, 95),
        'mean_of_file_means': np.mean(all_mean_values),
        'mean_of_file_stds': np.mean(all_std_values),
        'std_of_file_means': np.std(all_mean_values),
        'std_of_file_stds': np.std(all_std_values)
    }
    
    logger.info(f"Statistics calculated from {processed_files} files with {len(all_accelerations)} total samples")
    logger.info(f"Global acceleration range: {stats['global_min']:.3f} - {stats['global_max']:.3f}")
    logger.info(f"Global mean ± std: {stats['global_mean']:.3f} ± {stats['global_std']:.3f}")
    
    return stats

def calculate_intensity_thresholds(stats: dict, method: str = 'percentile') -> dict:
    """
    Calculate movement intensity thresholds based on acceleration statistics.
    
    Parameters:
    - stats: Dictionary containing acceleration statistics from analyze_acceleration_statistics
    - method: Method to calculate thresholds ('percentile', 'std', 'hybrid')
    
    Returns:
    - thresholds: dict with two boundaries:
        - 'low': boundary between low and medium
        - 'high': boundary between medium and high
    """
    if method == 'percentile':
        # Use 33rd and 66th percentiles for three-class boundaries
        thresholds = {
            'low': stats['percentile_33'],
            'high': stats['percentile_66']
        }
    elif method == 'std':
        # Use mean ± 0.5*std as boundaries
        mean = stats['global_mean']
        std = stats['global_std']
        thresholds = {
            'low': mean - 0.5 * std,
            'high': mean + 0.5 * std
        }
    elif method == 'hybrid':
        # Combine percentile and std approaches
        mean = stats['global_mean']
        std = stats['global_std']
        thresholds = {
            'low': min(stats['percentile_33'], mean - 0.5 * std),
            'high': max(stats['percentile_66'], mean + 0.5 * std)
        }
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    # Ensure thresholds are in ascending order
    if thresholds['high'] < thresholds['low']:
        # Adjust to maintain ordering
        thresholds['high'] = thresholds['low'] + 0.01

    logger.info(f"Intensity thresholds (three-class) using {method}: low={thresholds['low']:.3f}, high={thresholds['high']:.3f}")
    return thresholds

def classify_movement_intensity(accel_data: np.ndarray, thresholds: dict, aggregation: str = 'mean') -> tuple:
    """
    Classify movement intensity into three levels (low, medium, high) based on acceleration data and thresholds.
    
    Parameters:
    - accel_data: Array of acceleration magnitude values
    - thresholds: Dictionary with keys 'low' and 'high'
    - aggregation: Method to aggregate acceleration data ('mean', 'median', 'percentile_90', 'std')
    
    Returns:
    - Tuple containing (intensity_level, aggregated_value, confidence_score)
    """
    if aggregation == 'mean':
        agg_value = np.mean(accel_data)
    elif aggregation == 'median':
        agg_value = np.median(accel_data)
    elif aggregation == 'percentile_90':
        agg_value = np.percentile(accel_data, 90)
    elif aggregation == 'std':
        agg_value = np.std(accel_data)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Three-class classification
    if agg_value <= thresholds['low']:
        intensity = 'low'
    elif agg_value <= thresholds['high']:
        intensity = 'medium'
    else:
        intensity = 'high'
    
    # Confidence: distance to nearest boundary normalized by overall boundary span
    boundary_values = [thresholds['low'], thresholds['high']]
    min_distance = min(abs(agg_value - b) for b in boundary_values)
    max_range = thresholds['high'] - thresholds['low']
    confidence = min_distance / max_range if max_range > 0 else 1.0
    
    return intensity, agg_value, confidence

def analyze_movement_intensity(lab_file_path: str, thresholds: dict, side: str = 'right', 
                             aggregation: str = 'mean') -> dict:
    """
    Analyze movement intensity for a single lab data file.
    
    Parameters:
    - lab_file_path: Path to the lab data file (.joblib)
    - thresholds: Dictionary containing intensity thresholds
    - side: 'left' or 'right', specifies which leg's data to analyze
    - aggregation: Method to aggregate acceleration data
    
    Returns:
    - Dictionary containing intensity analysis results
    """
    logger.debug(f"Analyzing movement intensity for file: {lab_file_path}")
    
    # Load data
    df = load_lab_file(lab_file_path, side=side)
    accel_data = df['accel'].values
    
    # Classify intensity
    intensity, agg_value, confidence = classify_movement_intensity(accel_data, thresholds, aggregation)
    
    # Additional statistics
    results = {
        'file_path': lab_file_path,
        'side': side,
        'intensity_level': intensity,
        'aggregated_value': agg_value,
        'confidence_score': confidence,
        'aggregation_method': aggregation,
        'data_length': len(accel_data),
        'duration_seconds': len(accel_data) / LAB_SAMPLING_RATE,
        'mean_acceleration': np.mean(accel_data),
        'std_acceleration': np.std(accel_data),
        'min_acceleration': np.min(accel_data),
        'max_acceleration': np.max(accel_data),
        'percentile_90': np.percentile(accel_data, 90)
    }
    
    logger.debug(f"Movement intensity: {intensity} (confidence: {confidence:.3f})")
    logger.debug(f"Aggregated value ({aggregation}): {agg_value:.3f}")
    
    return results

def batch_analyze_movement_intensity(lab_data_dir: str, output_file: str = None, 
                                   side: str = 'right', aggregation: str = 'mean',
                                   threshold_method: str = 'percentile') -> pd.DataFrame:
    """
    Batch analyze movement intensity for all lab data files in a directory.

    @param lab_data_dir: Directory containing lab data files (.joblib)
    @param output_file: Optional path to save results as CSV
    @param side: 'left' or 'right', specifies which leg's data to analyze
    @param aggregation: Method to aggregate acceleration data ('mean', 'median', 'percentile_90', 'std')
    @param threshold_method: Method to calculate intensity thresholds ('percentile', 'std', 'hybrid')
    
    Parameters:
    - lab_data_dir: Directory containing lab data files
    - output_file: Optional path to save results as CSV
    - side: 'left' or 'right', specifies which leg's data to analyze
    - aggregation: Method to aggregate acceleration data
    - threshold_method: Method to calculate intensity thresholds
    
    Returns:
    - DataFrame containing analysis results for all files
    """
    logger.info(f"Starting batch analysis of directory: {lab_data_dir}")
    
    # First, analyze statistics to calculate thresholds
    stats = analyze_acceleration_statistics(lab_data_dir, side=side)
    thresholds = calculate_intensity_thresholds(stats, method=threshold_method)
    
    # Analyze each file
    results = []
    for filename in os.listdir(lab_data_dir):
        if filename.endswith('.joblib') and "DESKTOP" not in filename:
            file_path = os.path.join(lab_data_dir, filename)
            try:
                result = analyze_movement_intensity(file_path, thresholds, side=side, aggregation=aggregation)
                result['filename'] = filename
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze file {filename}: {str(e)}")
                continue
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Add summary information
    if len(df_results) > 0:
        logger.success(f"Batch Analysis Summary:")
        logger.success(f"Total files analyzed: {len(df_results)}")
        intensity_counts = df_results['intensity_level'].value_counts()
        for level, count in intensity_counts.items():
            logger.success(f"  {level}: {count} files ({count/len(df_results)*100:.1f}%)")
    
    # Save results if requested
    if output_file:
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
    
    return df_results

def main():
    """
    Main function to demonstrate movement intensity analysis.
    Modify the paths below according to your data location.
    """
    # Example usage - modify these paths according to your data
    lab_data_directory = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\OnTrack\L05\Baseline"
    
    try:
        # Method 1: Analyze a single file
        # single_file_path = os.path.join(lab_data_directory, "your_file.joblib")
        # if os.path.exists(single_file_path):
        #     stats = analyze_acceleration_statistics(lab_data_directory)
        #     thresholds = calculate_intensity_thresholds(stats, method='percentile')
        #     result = analyze_movement_intensity(single_file_path, thresholds)
        #     print(f"Single file analysis result: {result}")
        
        # Method 2: Batch analyze all files in directory
        if os.path.exists(lab_data_directory):
            logger.info("Starting batch analysis...")
            df_results = batch_analyze_movement_intensity(
                lab_data_dir=lab_data_directory,
                output_file=r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline\movement_intensity\movement_intensity_results.csv",
                side='right',
                aggregation='percentile_90',
                threshold_method='percentile'
            )
            if len(df_results) == 0:
                logger.error("No results obtained from analysis")
        else:
            print(f"Data directory not found: {lab_data_directory}")
            print("Please modify the lab_data_directory path in the main() function")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

