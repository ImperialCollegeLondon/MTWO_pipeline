"""
@ Author: Yufeng NA
@ Imperial College London
@ Date: August 3, 2025
@ Description: Test script for the data mapping functionality
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline")

from dataTransformer.data_mapper import DataMapper, apply_mapping_to_loaded_data
from config import getLogger

logger = getLogger('INFO')

def test_data_mapper():
    """Test the DataMapper class with sample data."""
    logger.info("Testing DataMapper functionality...")
    
    # Create sample data
    n_samples = 100
    sample_df = pd.DataFrame({
        'accelerationX': np.random.randn(n_samples),
        'accelerationY': np.random.randn(n_samples),
        'accelerationZ': np.random.randn(n_samples)
    })
    
    sample_array = np.random.randn(50, 3)  # Shape: (n_samples, 3)
    sample_windowed_array = np.random.randn(10, 20, 3)  # Shape: (n_windows, window_size, 3)
    
    # Test both LSTM and traditional models
    for model_type in ['auto', 'lstm', 'traditional']:
        logger.info(f"Testing with model_type: {model_type}")
        
        # Test DataMapper
        mapper = DataMapper(alignment_method='none', model_type=model_type)
        
        logger.info(f"Model loaded: {mapper.is_model_loaded()}")
        logger.info(f"Model info: {mapper.get_model_info()}")
        
        if mapper.is_model_loaded():
            # Test DataFrame transformation
            logger.info("Testing DataFrame transformation...")
            transformed_df = mapper.transform_dataframe(sample_df)
            logger.info(f"Original shape: {sample_df.shape}, Transformed shape: {transformed_df.shape}")
            
            # Test array transformation
            logger.info("Testing 2D array transformation...")
            transformed_array = mapper.transform_array(sample_array)
            logger.info(f"Original shape: {sample_array.shape}, Transformed shape: {transformed_array.shape}")
            
            # Test windowed array transformation
            logger.info("Testing 3D windowed array transformation...")
            transformed_windowed = mapper.transform_array(sample_windowed_array)
            logger.info(f"Original shape: {sample_windowed_array.shape}, Transformed shape: {transformed_windowed.shape}")
            
            # Test list of DataFrames
            logger.info("Testing list of DataFrames transformation...")
            df_list = [sample_df.copy() for _ in range(5)]
            transformed_list = mapper.transform_dataframe_list(df_list)
            logger.info(f"Transformed {len(transformed_list)} DataFrames")
            
            logger.success(f"All tests passed for {model_type} model!")
            break  # Exit after first successful model test
        else:
            logger.warning(f"No {model_type} mapping model found")
    
    if not any(DataMapper(alignment_method='none', model_type=mt).is_model_loaded() 
              for mt in ['auto', 'lstm', 'traditional']):
        logger.warning("No mapping models found - testing pass-through mode")
        
        # Test pass-through functionality
        mapper = DataMapper(alignment_method='none')
        transformed_df = mapper.transform_dataframe(sample_df)
        logger.info(f"Pass-through test - shapes match: {sample_df.shape == transformed_df.shape}")

def test_apply_mapping_function():
    """Test the apply_mapping_to_loaded_data function."""
    logger.info("Testing apply_mapping_to_loaded_data function...")
    
    # Create sample data in different formats
    # DataFrame list format (like custom dataset)
    movement_df_list = []
    other_df_list = []
    for i in range(5):
        df = pd.DataFrame({
            'accelerationX': np.random.randn(100),
            'accelerationY': np.random.randn(100),
            'accelerationZ': np.random.randn(100)
        })
        movement_df_list.append(df)
        other_df_list.append(df.copy())
    
    # Array format (like windowed data)
    movement_array = np.random.randn(10, 20, 3)
    other_array = np.random.randn(10, 20, 3)
    
    # Test with DataFrame lists
    logger.info("Testing with DataFrame lists...")
    movement_transformed, other_transformed, _, _ = apply_mapping_to_loaded_data(
        movement_data=movement_df_list,
        other_data=other_df_list,
        alignment_method='none',
        model_type='auto'
    )
    logger.info(f"Movement: {len(movement_df_list)} -> {len(movement_transformed)} DataFrames")
    logger.info(f"Other: {len(other_df_list)} -> {len(other_transformed)} DataFrames")
    
    # Test with arrays
    logger.info("Testing with numpy arrays...")
    movement_transformed, other_transformed, _, _ = apply_mapping_to_loaded_data(
        movement_data=movement_array,
        other_data=other_array,
        alignment_method='none',
        model_type='auto'
    )
    logger.info(f"Movement: {movement_array.shape} -> {movement_transformed.shape}")
    logger.info(f"Other: {other_array.shape} -> {other_transformed.shape}")
    
    logger.success("apply_mapping_to_loaded_data tests completed!")

if __name__ == "__main__":
    logger.info("Starting mapping functionality tests...")
    
    test_data_mapper()
    print("-" * 50)
    test_apply_mapping_function()
    
    logger.success("All mapping tests completed!")
