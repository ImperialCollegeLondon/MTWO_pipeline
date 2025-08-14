"""
@ Author: Yufeng NA
@ Imperial College London  
@ Date: August 3, 2025
@ Description: Data mapping module for applying trained mapping models to transform data
between different coordinate systems (e.g., Vicon Lab to Apple Watch).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Union, Optional
import sys
sys.path.insert(0, r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline")

# Try to import tensorflow for LSTM models, but don't fail if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

try:
    from config import getLogger
    logger = getLogger('INFO')
except ImportError:
    # Fallback logger if config import fails
    import logging
    logger = logging.getLogger(__name__)

class DataMapper:
    """
    A class to apply trained mapping models for data transformation between coordinate systems.
    """
    
    def __init__(self, model_path: Optional[str] = None, alignment_method: str = 'none', model_type: str = 'auto'):
        """
        Initialize the DataMapper with a trained mapping model.
        
        Parameters:
        - model_path: Path to the trained mapping model (.joblib file or directory for LSTM)
        - alignment_method: The alignment method used during training ('none', 'rotation_matrix', 'procrustes')
        - model_type: Type of model to load ('auto', 'lstm', 'traditional')
        """
        self.model = None
        self.alignment_method = alignment_method
        self.model_path = model_path
        self.model_type = model_type
        self.is_lstm_model = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Try to find the model in the mapping directory
            logger.error(f"Model path '{model_path}' does not exist. Attempting to find default model...")
            exit()
    
    
    def load_model(self, model_path: str):
        """
        Load a trained mapping model from file or directory.
        
        Parameters:
        - model_path: Path to the mapping model (.joblib file) or directory (for LSTM models)
        """
        try:
            if os.path.isdir(model_path):
                # Try to load LSTM model
                self._load_lstm_model(model_path)
            elif model_path.endswith('.joblib'):
                # Load traditional model
                self._load_traditional_model(model_path)
            else:
                logger.error(f"Unsupported model path format: {model_path}")
                exit()
                
        except Exception as e:
            logger.error(f"Failed to load mapping model from {model_path}: {str(e)}")
            exit()
    
    def _load_lstm_model(self, model_dir: str):
        """
        Load LSTM model from directory.
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is not available, cannot load LSTM model")
            exit()
        
        # Check if required files exist
        lstm_model_path = os.path.join(model_dir, 'lstm_model.h5')
        scaler_input_path = os.path.join(model_dir, 'scaler_input.joblib')
        scaler_output_path = os.path.join(model_dir, 'scaler_output.joblib')
        params_path = os.path.join(model_dir, 'model_params.pkl')
        
        if not all(os.path.exists(p) for p in [lstm_model_path, scaler_input_path, scaler_output_path, params_path]):
            logger.error(f"Missing required LSTM model files in {model_dir}")
            exit()

        # Load LSTM model components
        import pickle
        
        # Load model parameters
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        
        # Load the Keras model
        self.model = tf.keras.models.load_model(lstm_model_path)
        
        # Load scalers
        self.scaler_input = joblib.load(scaler_input_path)
        self.scaler_output = joblib.load(scaler_output_path)
        
        # Store parameters
        self.sequence_length = params.get('sequence_length', 50)
        self.lstm_units = params.get('lstm_units', 64)
        self.dropout_rate = params.get('dropout_rate', 0.2)
        
        self.model_path = model_dir
        self.is_lstm_model = True
        
        logger.info(f"LSTM mapping model loaded successfully from: {model_dir}")
        logger.info(f"Model parameters: sequence_length={self.sequence_length}, lstm_units={self.lstm_units}")
    
    def _load_traditional_model(self, model_path: str):
        """
        Load traditional (SVR/Linear) model from joblib file.
        """
        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.is_lstm_model = False
        
        logger.info(f"Traditional mapping model loaded successfully from: {model_path}")
        
        # Check if model is a list (multiple SVR models) or single model
        if isinstance(self.model, list):
            logger.info(f"Loaded {len(self.model)} SVR models for coordinate mapping")
        else:
            logger.info(f"Loaded single mapping model: {type(self.model).__name__}")
    
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a single DataFrame using the loaded mapping model.
        
        Parameters:
        - df: Input DataFrame with columns ['accelerationX', 'accelerationY', 'accelerationZ'] or 
              ['accelX', 'accelY', 'accelZ']
        
        Returns:
        - Transformed DataFrame with the same structure
        """
        if self.model is None:
            logger.debug("No mapping model loaded, returning original data")
            return df.copy()
        
        # Make a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Determine column names (handle different naming conventions)
        if 'accelerationX' in df.columns:
            accel_cols = ['accelerationX', 'accelerationY', 'accelerationZ']
        elif 'accelX' in df.columns:
            accel_cols = ['accelX', 'accelY', 'accelZ']
        else:
            logger.error("DataFrame must contain acceleration columns (accelerationX/Y/Z or accelX/Y/Z)")
            return df.copy()
        
        # Extract acceleration data
        accel_data = df[accel_cols].values
        
        try:
            if self.is_lstm_model:
                # Apply LSTM transformation
                transformed_accel = self._apply_lstm_transformation(accel_data)
            else:
                # Apply traditional transformation
                transformed_accel = self._apply_traditional_transformation(accel_data)
            
            # Update the DataFrame with transformed values
            transformed_df[accel_cols] = transformed_accel
            
            logger.debug(f"Applied mapping transformation to {len(df)} samples")
            
        except Exception as e:
            logger.error(f"Failed to apply mapping transformation: {str(e)}")
            return df.copy()
        
        return transformed_df
    
    def _apply_lstm_transformation(self, accel_data: np.ndarray) -> np.ndarray:
        """
        Apply LSTM model transformation to acceleration data.
        """
        # Scale input data
        accel_data_scaled = self.scaler_input.transform(accel_data)
        
        # Create sequences for LSTM
        if len(accel_data_scaled) < self.sequence_length:
            # If data is shorter than sequence length, pad with zeros or repeat
            logger.warning(f"Data length ({len(accel_data_scaled)}) is shorter than sequence length ({self.sequence_length}). Padding with zeros.")
            padded_data = np.zeros((self.sequence_length, accel_data_scaled.shape[1]))
            padded_data[:len(accel_data_scaled)] = accel_data_scaled
            accel_data_scaled = padded_data
        
        # Create sequences
        sequences = []
        for i in range(len(accel_data_scaled) - self.sequence_length + 1):
            sequences.append(accel_data_scaled[i:i + self.sequence_length])
        
        if not sequences:
            # Fallback: use the entire data as a single sequence
            sequences = [accel_data_scaled[-self.sequence_length:]]
        
        X_sequences = np.array(sequences)
        
        # Make predictions
        predictions_scaled = self.model.predict(X_sequences, verbose=0)
        
        # Inverse transform predictions
        predictions = self.scaler_output.inverse_transform(predictions_scaled)
        
        # Handle sequence output - we need to map back to original data length
        if len(predictions) != len(accel_data):
            # If we have fewer predictions due to sequence creation, extend or interpolate
            if len(predictions) < len(accel_data):
                # Pad with the last prediction
                last_prediction = predictions[-1]
                padding_needed = len(accel_data) - len(predictions)
                padding = np.tile(last_prediction, (padding_needed, 1))
                predictions = np.vstack([predictions, padding])
            else:
                # Truncate to match original length
                predictions = predictions[:len(accel_data)]
        
        return predictions
    
    def _apply_traditional_transformation(self, accel_data: np.ndarray) -> np.ndarray:
        """
        Apply traditional (SVR/Linear) model transformation to acceleration data.
        """
        if isinstance(self.model, list):
            # Multiple SVR models (one for each axis)
            transformed_accel = np.zeros_like(accel_data)
            for i, svr_model in enumerate(self.model):
                if i < accel_data.shape[1]:  # Safety check
                    transformed_accel[:, i] = svr_model.predict(accel_data)
        else:
            # Single model that outputs all axes
            transformed_accel = self.model.predict(accel_data)
        
        return transformed_accel
    
    def transform_array(self, data_array: np.ndarray) -> np.ndarray:
        """
        Transform a numpy array using the loaded mapping model.
        
        Parameters:
        - data_array: Input array with shape (n_samples, 3) or (n_windows, window_size, 3)
        
        Returns:
        - Transformed array with the same shape
        """
        if self.model is None:
            logger.debug("No mapping model loaded, returning original data")
            return data_array.copy()
        
        original_shape = data_array.shape
        
        # Handle different input shapes
        if len(original_shape) == 2:
            # Shape: (n_samples, 3)
            accel_data = data_array
        elif len(original_shape) == 3:
            # Shape: (n_windows, window_size, 3) - reshape to (n_samples, 3)
            accel_data = data_array.reshape(-1, original_shape[-1])
        else:
            logger.error(f"Unsupported array shape: {original_shape}")
            return data_array.copy()
        
        try:
            if self.is_lstm_model:
                # Apply LSTM transformation
                transformed_accel = self._apply_lstm_transformation(accel_data)
            else:
                # Apply traditional transformation
                transformed_accel = self._apply_traditional_transformation(accel_data)
            
            # Reshape back to original shape if needed
            if len(original_shape) == 3:
                transformed_accel = transformed_accel.reshape(original_shape)
            
            logger.debug(f"Applied mapping transformation to array with shape {original_shape}")
            
        except Exception as e:
            logger.error(f"Failed to apply mapping transformation to array: {str(e)}")
            return data_array.copy()
        
        return transformed_accel
    
    def transform_dataframe_list(self, df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Transform a list of DataFrames using the loaded mapping model.
        
        Parameters:
        - df_list: List of input DataFrames
        
        Returns:
        - List of transformed DataFrames
        """
        if self.model is None:
            logger.debug("No mapping model loaded, returning original data")
            return [df.copy() for df in df_list]
        
        transformed_list = []
        for i, df in enumerate(df_list):
            try:
                transformed_df = self.transform_dataframe(df)
                transformed_list.append(transformed_df)
            except Exception as e:
                logger.error(f"Failed to transform DataFrame {i}: {str(e)}")
                transformed_list.append(df.copy())  # Return original on error
        
        logger.info(f"Applied mapping transformation to {len(transformed_list)} DataFrames")
        return transformed_list
    
    def is_model_loaded(self) -> bool:
        """
        Check if a mapping model is currently loaded.
        
        Returns:
        - True if model is loaded, False otherwise
        """
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded mapping model.
        
        Returns:
        - Dictionary containing model information
        """
        info = {
            'model_loaded': self.is_model_loaded(),
            'model_path': self.model_path,
            'alignment_method': self.alignment_method,
            'model_type': 'LSTM' if self.is_lstm_model else 'Traditional'
        }
        
        if self.model is not None:
            if self.is_lstm_model:
                info['model_class'] = 'LSTM'
                info['sequence_length'] = getattr(self, 'sequence_length', 'Unknown')
                info['lstm_units'] = getattr(self, 'lstm_units', 'Unknown')
                info['dropout_rate'] = getattr(self, 'dropout_rate', 'Unknown')
                info['scaler_input'] = type(self.scaler_input).__name__ if hasattr(self, 'scaler_input') else 'None'
                info['scaler_output'] = type(self.scaler_output).__name__ if hasattr(self, 'scaler_output') else 'None'
            else:
                if isinstance(self.model, list):
                    info['model_class'] = 'Multiple SVR models'
                    info['num_models'] = len(self.model)
                else:
                    info['model_class'] = type(self.model).__name__
        
        return info


def apply_mapping_to_loaded_data(movement_data: Union[List[pd.DataFrame], np.ndarray], 
                                other_data: Union[List[pd.DataFrame], np.ndarray],
                                transport_data: Union[List[pd.DataFrame], np.ndarray] = None,
                                walking_data: Union[List[pd.DataFrame], np.ndarray] = None,
                                alignment_method: str = 'none',
                                model_path: Optional[str] = None,
                                model_type: str = 'auto') -> tuple:
    """
    Apply mapping transformation to loaded data.
    
    Parameters:
    - movement_data: Movement data (list of DataFrames or numpy array)
    - other_data: Other activity data (list of DataFrames or numpy array)
    - transport_data: Transport data (optional)
    - walking_data: Walking data (optional)
    - alignment_method: Alignment method used during training
    - model_path: Path to the mapping model (optional)
    - model_type: Type of model to load ('auto', 'lstm', 'traditional')
    
    Returns:
    - Tuple of transformed data in the same format as input
    """
    # Initialize the mapper
    mapper = DataMapper(model_path=model_path, alignment_method=alignment_method, model_type=model_type)
    
    if not mapper.is_model_loaded():
        logger.warning("No mapping model loaded. Data will be returned unchanged.")
        return movement_data, other_data, transport_data, walking_data
    
    logger.info("Applying mapping transformation to loaded data...")
    model_info = mapper.get_model_info()
    logger.info(f"Using model: {model_info}")
    
    # Transform movement data
    if isinstance(movement_data, list):
        movement_transformed = mapper.transform_dataframe_list(movement_data)
    else:
        movement_transformed = mapper.transform_array(movement_data)
    
    # Transform other data
    if isinstance(other_data, list):
        other_transformed = mapper.transform_dataframe_list(other_data)
    else:
        other_transformed = mapper.transform_array(other_data)
    
    # Transform transport data if provided
    transport_transformed = None
    if transport_data is not None:
        if isinstance(transport_data, list):
            transport_transformed = mapper.transform_dataframe_list(transport_data)
        else:
            transport_transformed = mapper.transform_array(transport_data)
    
    # Transform walking data if provided
    walking_transformed = None
    if walking_data is not None:
        if isinstance(walking_data, list):
            walking_transformed = mapper.transform_dataframe_list(walking_data)
        else:
            walking_transformed = mapper.transform_array(walking_data)
    
    logger.success("Mapping transformation completed successfully!")
    
    return movement_transformed, other_transformed, transport_transformed, walking_transformed


if __name__ == "__main__":
    # Example usage and testing
    print("Testing DataMapper...")
    
    # Create a sample DataFrame for testing
    sample_data = pd.DataFrame({
        'accelerationX': np.random.randn(100),
        'accelerationY': np.random.randn(100), 
        'accelerationZ': np.random.randn(100)
    })
    
    # Test the mapper
    mapper = DataMapper()
    
    if mapper.is_model_loaded():
        print("Model loaded successfully!")
        print("Model info:", mapper.get_model_info())
        
        # Test transformation
        transformed_data = mapper.transform_dataframe(sample_data)
        print(f"Original data shape: {sample_data.shape}")
        print(f"Transformed data shape: {transformed_data.shape}")
        
        print("DataMapper test completed!")
    else:
        print("No model found for testing.")
