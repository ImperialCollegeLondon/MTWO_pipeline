import sys
sys.path.insert(0, r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline")

import os
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt
from typing import List, Dict, Tuple

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
    This function remains the same as the original to maintain consistent threshold calculation.
    
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

def segment_data(accel_data: np.ndarray, timestamps: np.ndarray, segment_duration: float = 10.0) -> List[Tuple[np.ndarray, float, float]]:
    """
    Segment acceleration data into time windows.
    
    Parameters:
    - accel_data: Array of acceleration magnitude values
    - timestamps: Array of timestamps corresponding to acceleration data
    - segment_duration: Duration of each segment in seconds
    
    Returns:
    - List of tuples, each containing (segment_accel_data, start_time, end_time)
    """
    segments = []
    total_duration = timestamps[-1] - timestamps[0]
    num_segments = int(np.ceil(total_duration / segment_duration))
    
    logger.debug(f"Segmenting data: total duration={total_duration:.2f}s, segment_duration={segment_duration}s, num_segments={num_segments}")
    
    for i in range(num_segments):
        start_time = timestamps[0] + i * segment_duration
        end_time = min(timestamps[0] + (i + 1) * segment_duration, timestamps[-1])
        
        # Find indices for this time segment
        start_idx = np.searchsorted(timestamps, start_time, side='left')
        end_idx = np.searchsorted(timestamps, end_time, side='right')
        
        # Extract segment data
        if end_idx > start_idx:  # Ensure we have data in this segment
            segment_accel = accel_data[start_idx:end_idx]
            segments.append((segment_accel, start_time, end_time))
            logger.debug(f"Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s, {len(segment_accel)} samples")
    
    return segments

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
    if len(accel_data) == 0:
        return 'unknown', 0.0, 0.0
    
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

def analyze_movement_intensity_by_segments(lab_file_path: str, thresholds: dict, side: str = 'right', 
                                         aggregation: str = 'mean', segment_duration: float = 10.0) -> List[Dict]:
    """
    Analyze movement intensity for a single lab data file by segmenting it into time windows.
    
    Parameters:
    - lab_file_path: Path to the lab data file (.joblib)
    - thresholds: Dictionary containing intensity thresholds
    - side: 'left' or 'right', specifies which leg's data to analyze
    - aggregation: Method to aggregate acceleration data
    - segment_duration: Duration of each segment in seconds
    
    Returns:
    - List of dictionaries containing intensity analysis results for each segment
    """
    logger.debug(f"Analyzing movement intensity by segments for file: {lab_file_path}")
    
    # Load data
    df = load_lab_file(lab_file_path, side=side)
    accel_data = df['accel'].values
    timestamps = df['timestamp'].values
    
    # Segment the data
    segments = segment_data(accel_data, timestamps, segment_duration)
    
    results = []
    for i, (segment_accel, start_time, end_time) in enumerate(segments):
        # Classify intensity for this segment
        intensity, agg_value, confidence = classify_movement_intensity(segment_accel, thresholds, aggregation)
        
        # Calculate segment statistics
        segment_result = {
            'file_path': lab_file_path,
            'filename': os.path.basename(lab_file_path),
            'segment_id': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'side': side,
            'intensity_level': intensity,
            'aggregated_value': agg_value,
            'confidence_score': confidence,
            'aggregation_method': aggregation,
            'data_length': len(segment_accel),
            'mean_acceleration': np.mean(segment_accel) if len(segment_accel) > 0 else 0,
            'std_acceleration': np.std(segment_accel) if len(segment_accel) > 0 else 0,
            'min_acceleration': np.min(segment_accel) if len(segment_accel) > 0 else 0,
            'max_acceleration': np.max(segment_accel) if len(segment_accel) > 0 else 0,
            'percentile_90': np.percentile(segment_accel, 90) if len(segment_accel) > 0 else 0
        }
        
        results.append(segment_result)
        
        logger.debug(f"Segment {i+1} ({start_time:.1f}-{end_time:.1f}s): {intensity} intensity (confidence: {confidence:.3f})")
    
    return results

def batch_analyze_movement_intensity_by_segments(lab_data_dir: str, output_file: str = None, 
                                               side: str = 'right', aggregation: str = 'mean',
                                               threshold_method: str = 'percentile', 
                                               segment_duration: float = 10.0) -> pd.DataFrame:
    """
    Batch analyze movement intensity for all lab data files in a directory using segmentation.
    
    Parameters:
    - lab_data_dir: Directory containing lab data files (.joblib)
    - output_file: Optional path to save results as CSV
    - side: 'left' or 'right', specifies which leg's data to analyze
    - aggregation: Method to aggregate acceleration data ('mean', 'median', 'percentile_90', 'std')
    - threshold_method: Method to calculate intensity thresholds ('percentile', 'std', 'hybrid')
    - segment_duration: Duration of each segment in seconds
    
    Returns:
    - DataFrame containing analysis results for all segments from all files
    """
    logger.info(f"Starting batch analysis with segmentation of directory: {lab_data_dir}")
    logger.info(f"Segment duration: {segment_duration} seconds")
    
    # First, analyze statistics to calculate thresholds (same as original method)
    stats = analyze_acceleration_statistics(lab_data_dir, side=side)
    thresholds = calculate_intensity_thresholds(stats, method=threshold_method)
    
    # Analyze each file by segments
    all_results = []
    processed_files = 0
    total_segments = 0
    
    for filename in os.listdir(lab_data_dir):
        if filename.endswith('.joblib') and "DESKTOP" not in filename:
            file_path = os.path.join(lab_data_dir, filename)
            try:
                segments_results = analyze_movement_intensity_by_segments(
                    file_path, thresholds, side=side, aggregation=aggregation, 
                    segment_duration=segment_duration
                )
                all_results.extend(segments_results)
                processed_files += 1
                total_segments += len(segments_results)
                logger.debug(f"File {filename}: {len(segments_results)} segments analyzed")
                
            except Exception as e:
                logger.warning(f"Failed to analyze file {filename}: {str(e)}")
                continue
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Add summary information
    if len(df_results) > 0:
        logger.success(f"Segmented Analysis Summary:")
        logger.success(f"Total files processed: {processed_files}")
        logger.success(f"Total segments analyzed: {total_segments}")
        logger.success(f"Average segments per file: {total_segments/processed_files:.1f}")
        
        intensity_counts = df_results['intensity_level'].value_counts()
        for level, count in intensity_counts.items():
            logger.success(f"  {level}: {count} segments ({count/len(df_results)*100:.1f}%)")
        
        # File-level summary (majority voting)
        file_summary = []
        for filename in df_results['filename'].unique():
            file_segments = df_results[df_results['filename'] == filename]
            majority_intensity = file_segments['intensity_level'].mode().iloc[0]
            confidence_mean = file_segments['confidence_score'].mean()
            
            file_summary.append({
                'filename': filename,
                'total_segments': len(file_segments),
                'majority_intensity': majority_intensity,
                'average_confidence': confidence_mean,
                'low_segments': sum(file_segments['intensity_level'] == 'low'),
                'medium_segments': sum(file_segments['intensity_level'] == 'medium'),
                'high_segments': sum(file_segments['intensity_level'] == 'high')
            })
        
        df_file_summary = pd.DataFrame(file_summary)
        logger.success("File-level summary (majority voting):")
        file_intensity_counts = df_file_summary['majority_intensity'].value_counts()
        for level, count in file_intensity_counts.items():
            logger.success(f"  {level}: {count} files ({count/len(df_file_summary)*100:.1f}%)")
    
    # Save results if requested
    if output_file:
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
        
        # Also save file-level summary
        if len(df_results) > 0:
            summary_file = output_file.replace('.csv', '_file_summary.csv')
            df_file_summary.to_csv(summary_file, index=False)
            logger.info(f"File-level summary saved to: {summary_file}")
    
    return df_results

def compare_methods(lab_data_dir: str, side: str = 'right', aggregation: str = 'mean',
                   threshold_method: str = 'percentile', segment_duration: float = 10.0) -> Dict:
    """
    Compare results between the original method (whole file analysis) and the new segmented method.
    
    Parameters:
    - lab_data_dir: Directory containing lab data files (.joblib)
    - side: 'left' or 'right', specifies which leg's data to analyze
    - aggregation: Method to aggregate acceleration data
    - threshold_method: Method to calculate intensity thresholds
    - segment_duration: Duration of each segment in seconds
    
    Returns:
    - Dictionary containing comparison results
    """
    logger.info("Comparing original method vs segmented method")
    
    # Get statistics and thresholds
    stats = analyze_acceleration_statistics(lab_data_dir, side=side)
    thresholds = calculate_intensity_thresholds(stats, method=threshold_method)
    
    original_results = []
    segmented_results = []
    
    for filename in os.listdir(lab_data_dir):
        if filename.endswith('.joblib') and "DESKTOP" not in filename:
            file_path = os.path.join(lab_data_dir, filename)
            try:
                # Original method: analyze whole file
                df = load_lab_file(file_path, side=side)
                accel_data = df['accel'].values
                intensity, agg_value, confidence = classify_movement_intensity(accel_data, thresholds, aggregation)
                
                original_results.append({
                    'filename': filename,
                    'method': 'original',
                    'intensity': intensity,
                    'confidence': confidence,
                    'aggregated_value': agg_value
                })
                
                # Segmented method
                segments_results = analyze_movement_intensity_by_segments(
                    file_path, thresholds, side=side, aggregation=aggregation, 
                    segment_duration=segment_duration
                )
                
                # Use majority voting for file-level classification
                intensities = [r['intensity_level'] for r in segments_results]
                majority_intensity = max(set(intensities), key=intensities.count)
                avg_confidence = np.mean([r['confidence_score'] for r in segments_results])
                avg_aggregated_value = np.mean([r['aggregated_value'] for r in segments_results])
                
                segmented_results.append({
                    'filename': filename,
                    'method': 'segmented',
                    'intensity': majority_intensity,
                    'confidence': avg_confidence,
                    'aggregated_value': avg_aggregated_value,
                    'num_segments': len(segments_results)
                })
                
            except Exception as e:
                logger.warning(f"Failed to compare methods for file {filename}: {str(e)}")
                continue
    
    # Calculate comparison statistics
    comparison = {
        'original_results': original_results,
        'segmented_results': segmented_results,
        'total_files': len(original_results)
    }
    
    if len(original_results) > 0:
        # Count agreements and disagreements
        agreements = 0
        disagreements = []
        
        for orig, seg in zip(original_results, segmented_results):
            if orig['intensity'] == seg['intensity']:
                agreements += 1
            else:
                disagreements.append({
                    'filename': orig['filename'],
                    'original_intensity': orig['intensity'],
                    'segmented_intensity': seg['intensity'],
                    'original_confidence': orig['confidence'],
                    'segmented_confidence': seg['confidence']
                })
        
        comparison['agreements'] = agreements
        comparison['disagreements'] = disagreements
        comparison['agreement_rate'] = agreements / len(original_results)
        
        logger.success(f"Method Comparison Summary:")
        logger.success(f"Total files compared: {len(original_results)}")
        logger.success(f"Agreements: {agreements} ({comparison['agreement_rate']*100:.1f}%)")
        logger.success(f"Disagreements: {len(disagreements)} ({(1-comparison['agreement_rate'])*100:.1f}%)")
        
        if disagreements:
            logger.info("Files with disagreements:")
            for d in disagreements:
                logger.info(f"  {d['filename']}: {d['original_intensity']} -> {d['segmented_intensity']}")
    
    return comparison

def main():
    """
    Main function to demonstrate segmented movement intensity analysis.
    """
    # Configuration
    lab_data_directory = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\Data\OnTrack\L05\Baseline"
    output_dir = r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline\movement_intensity"
    
    try:
        if os.path.exists(lab_data_directory):
            logger.info("Starting segmented analysis...")
            
            # Method 1: Segmented analysis
            df_results = batch_analyze_movement_intensity_by_segments(
                lab_data_dir=lab_data_directory,
                output_file=os.path.join(output_dir, "movement_intensity_segmented_results.csv"),
                side='right',
                aggregation='mean',
                threshold_method='percentile',
                segment_duration=3  # 10-second segments
            )
            
            if len(df_results) == 0:
                logger.error("No results obtained from segmented analysis")
            else:
                logger.success(f"Segmented analysis completed successfully with {len(df_results)} segments")
            
            # Method 2: Compare methods
            logger.info("Comparing original vs segmented methods...")
            comparison = compare_methods(
                lab_data_dir=lab_data_directory,
                side='right',
                aggregation='mean',
                threshold_method='percentile',
                segment_duration=3
            )
            
            # Save comparison results
            comparison_file = os.path.join(output_dir, "method_comparison_results_3.csv")
            comparison_df = pd.DataFrame(comparison['original_results'] + comparison['segmented_results'])
            comparison_df.to_csv(comparison_file, index=False)
            logger.info(f"Method comparison saved to: {comparison_file}")
            
        else:
            print(f"Data directory not found: {lab_data_directory}")
            print("Please modify the lab_data_directory path in the main() function")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print('------------------------------')
    main()
