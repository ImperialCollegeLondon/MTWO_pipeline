import os
import sys
sys.path.insert(0, r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle
from loguru import logger
from config import getLogger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# 导入现有的数据加载和处理函数
from mapping import (
    load_paired_training_data, 
    convert_sample_to_dataframes,
    apply_rotation_matrix,
    align_coordinate_systems_procrustes,
    calculate_rotation_matrix_from_sample
)

logger = getLogger('INFO')

class LSTMMappingModel:
    """
    LSTM-based mapping model for converting Vicon acceleration data to Apple Watch acceleration data
    """
    
    def __init__(self, sequence_length=50, lstm_units=64, dropout_rate=0.2):
        """
        Initialize LSTM mapping model
        
        @param sequence_length: Length of input sequences for LSTM
        @param lstm_units: Number of LSTM units in each layer
        @param dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler_input = None
        self.scaler_output = None
        self.training_history = None
        
    def create_sequences(self, data, sequence_length):
        """
        Create sequences for LSTM training
        
        @param data: Input data array (n_samples, n_features)
        @param sequence_length: Length of each sequence
        @return: Sequences array (n_sequences, sequence_length, n_features)
        """
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def prepare_data(self, vicon_data_list, aw_data_list):
        """
        Prepare data for LSTM training
        
        @param vicon_data_list: List of Vicon DataFrames
        @param aw_data_list: List of Apple Watch DataFrames
        @return: Tuple of (X_sequences, y_sequences, scalers)
        """
        logger.info("Preparing data for LSTM training...")
        
        # Combine all data
        all_vicon_data = []
        all_aw_data = []
        
        for vicon_df, aw_df in zip(vicon_data_list, aw_data_list):
            vicon_features = vicon_df[['accelX', 'accelY', 'accelZ']].values
            aw_targets = aw_df[['accelX', 'accelY', 'accelZ']].values
            
            all_vicon_data.append(vicon_features)
            all_aw_data.append(aw_targets)
        
        # Stack all data
        X_combined = np.vstack(all_vicon_data)
        y_combined = np.vstack(all_aw_data)
        
        # Scale the data
        self.scaler_input = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_output = MinMaxScaler(feature_range=(-1, 1))
        
        X_scaled = self.scaler_input.fit_transform(X_combined)
        y_scaled = self.scaler_output.fit_transform(y_combined)
        
        logger.info(f"Combined data shape: X={X_combined.shape}, y={y_combined.shape}")
        
        # Create sequences
        X_sequences = self.create_sequences(X_scaled, self.sequence_length)
        y_sequences = self.create_sequences(y_scaled, self.sequence_length)
        
        # For LSTM, we typically predict the next value, so align sequences
        X_sequences = X_sequences[:-1]  # Remove last sequence
        y_sequences = y_sequences[1:]   # Remove first sequence
        
        # Take only the last timestep of each target sequence for prediction
        y_sequences = y_sequences[:, -1, :]  # Shape: (n_sequences, n_features)
        
        logger.info(f"Sequence data shape: X={X_sequences.shape}, y={y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        @param input_shape: Shape of input sequences (sequence_length, n_features)
        @return: Compiled LSTM model
        """
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(3, activation='linear')  # Output 3 acceleration components
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_sequences, y_sequences, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the LSTM model
        
        @param X_sequences: Input sequences
        @param y_sequences: Target sequences
        @param validation_split: Fraction of data to use for validation
        @param epochs: Number of training epochs
        @param batch_size: Training batch size
        @return: Training history
        """
        logger.info("Building and training LSTM model...")
        
        # Build model
        input_shape = (X_sequences.shape[1], X_sequences.shape[2])
        self.model = self.build_model(input_shape)
        
        # Print model summary
        logger.info("LSTM Model Architecture:")
        self.model.summary()
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        self.training_history = self.model.fit(
            X_sequences, y_sequences,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("LSTM model training completed!")
        return self.training_history
    
    def predict(self, X_sequences):
        """
        Make predictions using the trained LSTM model
        
        @param X_sequences: Input sequences for prediction
        @return: Predicted acceleration values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        predictions_scaled = self.model.predict(X_sequences)
        predictions = self.scaler_output.inverse_transform(predictions_scaled)
        
        return predictions
    
    def evaluate(self, X_sequences, y_sequences):
        """
        Evaluate the trained model
        
        @param X_sequences: Input sequences
        @param y_sequences: True target values
        @return: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_sequences)
        
        # Inverse transform to original scale
        y_pred = self.scaler_output.inverse_transform(y_pred_scaled)
        y_true_scaled = self.scaler_output.transform(y_sequences)
        y_true = self.scaler_output.inverse_transform(y_true_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Per-axis metrics
        axis_names = ['X', 'Y', 'Z']
        axis_metrics = {}
        for i, axis in enumerate(axis_names):
            axis_rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            axis_r2 = r2_score(y_true[:, i], y_pred[:, i])
            axis_metrics[f'{axis}_rmse'] = axis_rmse
            axis_metrics[f'{axis}_r2'] = axis_r2
        
        metrics = {
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'n_samples': len(y_sequences),
            **axis_metrics
        }
        
        return metrics
    
    def save_model(self, model_dir):
        """
        Save the trained LSTM model and scalers
        
        @param model_dir: Directory to save the model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the Keras model
        model_path = os.path.join(model_dir, 'lstm_model.h5')
        self.model.save(model_path)
        
        # Save scalers
        scaler_input_path = os.path.join(model_dir, 'scaler_input.joblib')
        scaler_output_path = os.path.join(model_dir, 'scaler_output.joblib')
        joblib.dump(self.scaler_input, scaler_input_path)
        joblib.dump(self.scaler_output, scaler_output_path)
        
        # Save model parameters
        params = {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate
        }
        params_path = os.path.join(model_dir, 'model_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
        
        # Save training history if available
        if self.training_history is not None:
            history_path = os.path.join(model_dir, 'training_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.training_history.history, f)
        
        logger.info(f"LSTM model saved to {model_dir}")
    
    def load_model(self, model_dir):
        """
        Load a pre-trained LSTM model
        
        @param model_dir: Directory containing the saved model
        """
        # Load model parameters
        params_path = os.path.join(model_dir, 'model_params.pkl')
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        
        self.sequence_length = params['sequence_length']
        self.lstm_units = params['lstm_units']
        self.dropout_rate = params['dropout_rate']
        
        # Load the Keras model
        model_path = os.path.join(model_dir, 'lstm_model.h5')
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scalers
        scaler_input_path = os.path.join(model_dir, 'scaler_input.joblib')
        scaler_output_path = os.path.join(model_dir, 'scaler_output.joblib')
        self.scaler_input = joblib.load(scaler_input_path)
        self.scaler_output = joblib.load(scaler_output_path)
        
        logger.info(f"LSTM model loaded from {model_dir}")


def process_paired_samples_with_alignment_lstm(training_samples, alignment_method='rotation_matrix'):
    """
    Process paired samples for LSTM training with coordinate system alignment
    
    @param training_samples: List of paired training samples
    @param alignment_method: Method for alignment ('rotation_matrix', 'procrustes', or 'none')
    @return: Tuple of (aligned_aw_data_list, aligned_vicon_data_list, alignment_info)
    """
    aligned_aw_data = []
    aligned_vicon_data = []
    alignment_info = {
        'method': alignment_method,
        'rotation_matrices': [],
        'sample_stats': []
    }
    
    logger.info(f"Processing {len(training_samples)} paired samples for LSTM with alignment method: {alignment_method}")
    
    for i, sample in enumerate(training_samples):
        try:
            # Convert sample to DataFrames
            aw_df, vicon_df = convert_sample_to_dataframes(sample)
            
            # Apply coordinate system alignment
            if alignment_method == 'rotation_matrix':
                # Use predefined rotation matrix
                rotation_matrix = np.array([[ 0.58377147,  0.28237329,  0.76123334],
                                            [ 0.36897715,  0.74289845, -0.55853179],
                                            [-0.72323352,  0.60693263,  0.32949362]])
                
                aligned_vicon_df = apply_rotation_matrix(vicon_df, rotation_matrix)
                aligned_aw_df = aw_df.copy()
                alignment_info['rotation_matrices'].append(rotation_matrix)
                
            elif alignment_method == 'procrustes':
                aligned_vicon_df, rotation_matrix = align_coordinate_systems_procrustes(vicon_df, aw_df)
                aligned_aw_df = aw_df.copy()
                alignment_info['rotation_matrices'].append(rotation_matrix)
                
            elif alignment_method == 'none':
                aligned_vicon_df = vicon_df.copy()
                aligned_aw_df = aw_df.copy()
                alignment_info['rotation_matrices'].append(np.eye(3))
            
            else:
                raise ValueError(f"Unknown alignment method: {alignment_method}")
            
            aligned_aw_data.append(aligned_aw_df)
            aligned_vicon_data.append(aligned_vicon_df)
            
            # Record sample statistics
            sample_stats = {
                'sample_id': sample['sample_id'],
                'filename_aw': sample['filename_aw'],
                'filename_vicon': sample['filename_vicon'],
                'length': len(aligned_aw_df),
                'aw_mean_accel': np.mean([aligned_aw_df['accelX'].mean(), 
                                        aligned_aw_df['accelY'].mean(), 
                                        aligned_aw_df['accelZ'].mean()]),
                'vicon_mean_accel': np.mean([aligned_vicon_df['accelX'].mean(), 
                                           aligned_vicon_df['accelY'].mean(), 
                                           aligned_vicon_df['accelZ'].mean()])
            }
            alignment_info['sample_stats'].append(sample_stats)
            
            logger.debug(f"Processed sample {i+1}/{len(training_samples)}: {sample['filename_aw']}")
            
        except Exception as e:
            logger.error(f"Error processing sample {i+1} ({sample.get('filename_aw', 'unknown')}): {str(e)}")
            continue
    
    logger.info(f"Successfully processed {len(aligned_aw_data)} samples with {alignment_method} alignment for LSTM")
    return aligned_aw_data, aligned_vicon_data, alignment_info


def train_lstm_mapping_model(training_samples, alignment_method='rotation_matrix', 
                           sequence_length=50, lstm_units=64, dropout_rate=0.2,
                           epochs=100, batch_size=32):
    """
    Train LSTM mapping model from paired training data
    
    @param training_samples: List of paired training samples
    @param alignment_method: Method for coordinate system alignment
    @param sequence_length: Length of input sequences for LSTM
    @param lstm_units: Number of LSTM units
    @param dropout_rate: Dropout rate for regularization
    @param epochs: Number of training epochs
    @param batch_size: Training batch size
    @return: Tuple of (trained_model, alignment_info, training_metrics)
    """
    logger.info("Starting LSTM mapping model training from paired data...")
    
    # Process samples with alignment
    aligned_aw_data, aligned_vicon_data, alignment_info = process_paired_samples_with_alignment_lstm(
        training_samples, alignment_method
    )
    
    if not aligned_aw_data or not aligned_vicon_data:
        raise ValueError("No successfully aligned data found for LSTM training")
    
    # Create LSTM model
    lstm_model = LSTMMappingModel(
        sequence_length=sequence_length,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    
    # Prepare data for LSTM training
    X_sequences, y_sequences = lstm_model.prepare_data(aligned_vicon_data, aligned_aw_data)
    
    # Train the model
    training_history = lstm_model.train(
        X_sequences, y_sequences,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate the model
    metrics = lstm_model.evaluate(X_sequences, y_sequences)
    
    logger.info(f"LSTM mapping model training completed:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  Training sequences: {len(X_sequences)}")
    
    return lstm_model, alignment_info, metrics


def visualize_lstm_training_history(history, save_path=None):
    """
    Visualize LSTM training history
    
    @param history: Training history from Keras model
    @param save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training MAE
    ax2.plot(history.history['mae'], label='Training MAE', color='blue')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
    ax2.set_title('Model MAE During Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def test_lstm_model_on_sample(lstm_model, test_sample, alignment_method='rotation_matrix'):
    """
    Test LSTM model on a single sample
    
    @param lstm_model: Trained LSTM model
    @param test_sample: Test sample dictionary
    @param alignment_method: Alignment method used during training
    @return: Prediction results and metrics
    """
    # Convert sample to DataFrames
    aw_df, vicon_df = convert_sample_to_dataframes(test_sample)
    
    # Apply same alignment as training
    if alignment_method == 'rotation_matrix':
        rotation_matrix = np.array([[ 0.58377147,  0.28237329,  0.76123334],
                                    [ 0.36897715,  0.74289845, -0.55853179],
                                    [-0.72323352,  0.60693263,  0.32949362]])
        aligned_vicon_df = apply_rotation_matrix(vicon_df, rotation_matrix)
    elif alignment_method == 'procrustes':
        aligned_vicon_df, _ = align_coordinate_systems_procrustes(vicon_df, aw_df)
    else:
        aligned_vicon_df = vicon_df.copy()
    
    # Prepare input sequences
    vicon_data = aligned_vicon_df[['accelX', 'accelY', 'accelZ']].values
    vicon_scaled = lstm_model.scaler_input.transform(vicon_data)
    
    # Create sequences
    X_test_sequences = lstm_model.create_sequences(vicon_scaled, lstm_model.sequence_length)
    
    if len(X_test_sequences) == 0:
        logger.warning(f"Sample too short for sequence length {lstm_model.sequence_length}")
        return None, None
    
    # Make predictions
    y_pred = lstm_model.predict(X_test_sequences)
    
    # Get true values (aligned with predictions)
    aw_true = aw_df[['accelX', 'accelY', 'accelZ']].values[lstm_model.sequence_length:]
    
    # Calculate metrics
    if len(aw_true) != len(y_pred):
        min_len = min(len(aw_true), len(y_pred))
        aw_true = aw_true[:min_len]
        y_pred = y_pred[:min_len]
    
    test_rmse = np.sqrt(mean_squared_error(aw_true, y_pred))
    test_r2 = r2_score(aw_true, y_pred)
    test_mae = np.mean(np.abs(aw_true - y_pred))
    
    metrics = {
        'rmse': test_rmse,
        'r2_score': test_r2,
        'mae': test_mae,
        'sample_length': len(y_pred)
    }
    
    results = {
        'true_values': aw_true,
        'predictions': y_pred,
        'metrics': metrics
    }
    
    return results, metrics


if __name__ == "__main__":
    print("=== LSTM映射模型训练 ===")
    
    # --- 1. 加载配对训练数据 ---
    paired_data_path = "paired_training_data/paired_training_data.pkl"
    
    try:
        training_samples = load_paired_training_data(paired_data_path)
        print(f"成功加载 {len(training_samples)} 个配对训练样本")
        
        # 显示样本信息
        print("\n配对样本概览:")
        for i, sample in enumerate(training_samples[:5]):
            print(f"  样本 {sample['sample_id']:2d}: {sample['filename_aw']:25s} <-> {sample['filename_vicon']:25s} "
                  f"({sample['length']:4d} 数据点, {sample['duration']:.1f}秒)")
        if len(training_samples) > 5:
            print(f"  ... 还有 {len(training_samples) - 5} 个样本")
    
    except FileNotFoundError:
        print(f"错误: 找不到配对训练数据文件 {paired_data_path}")
        print("请先运行 build_training_set.py 生成配对训练数据")
        exit(1)
    
    # --- 2. LSTM模型训练参数 ---
    lstm_params = {
        'sequence_length': 50,    # 序列长度
        'lstm_units': 64,        # LSTM单元数
        'dropout_rate': 0.2,     # Dropout率
        'epochs': 100,           # 训练轮数
        'batch_size': 32         # 批次大小
    }
    
    print(f"\nLSTM模型参数:")
    for key, value in lstm_params.items():
        print(f"  {key}: {value}")
    
    # --- 3. 训练不同对齐方法的LSTM模型 ---
    alignment_methods = ['rotation_matrix', 'procrustes', 'none']
    lstm_results = {}
    
    for method in alignment_methods:
        print(f"\n--- 使用 {method} 对齐方法训练LSTM模型 ---")
        
        try:
            # 训练LSTM模型
            lstm_model, alignment_info, metrics = train_lstm_mapping_model(
                training_samples, 
                alignment_method=method,
                **lstm_params
            )
            
            # 保存模型
            model_dir = f"lstm_mapping_models_{method}"
            lstm_model.save_model(model_dir)
            
            # 保存对齐信息
            alignment_path = os.path.join(model_dir, 'alignment_info.pkl')
            with open(alignment_path, 'wb') as f:
                pickle.dump(alignment_info, f)
            
            # 保存训练指标
            metrics_path = os.path.join(model_dir, 'training_metrics.pkl')
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            
            lstm_results[method] = {
                'model': lstm_model,
                'metrics': metrics,
                'model_dir': model_dir
            }
            
            print(f"LSTM模型已保存到 {model_dir}/ 目录")
            print(f"{method} 方法LSTM训练结果:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R² Score: {metrics['r2_score']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  训练序列数: {metrics['n_samples']}")
            
            # 可视化训练历史
            if lstm_model.training_history:
                history_plot_path = os.path.join(model_dir, 'training_history.png')
                visualize_lstm_training_history(lstm_model.training_history, history_plot_path)
            
        except Exception as e:
            print(f"使用 {method} 方法训练LSTM时出错: {str(e)}")
            continue
    
    # --- 4. 模型性能比较 ---
    if lstm_results:
        print(f"\n--- LSTM模型性能比较 ---")
        print(f"{'方法':<15} {'RMSE':<10} {'R² Score':<10} {'MAE':<10} {'序列数':<10}")
        print("-" * 60)
        
        best_method = None
        best_r2 = -float('inf')
        
        for method, result in lstm_results.items():
            metrics = result['metrics']
            print(f"{method:<15} {metrics['rmse']:<10.4f} {metrics['r2_score']:<10.4f} "
                  f"{metrics['mae']:<10.4f} {metrics['n_samples']:<10}")
            
            if metrics['r2_score'] > best_r2:
                best_r2 = metrics['r2_score']
                best_method = method
        
        print(f"\n最佳LSTM方法: {best_method} (R² = {best_r2:.4f})")
        
        # 创建性能比较图
        methods = list(lstm_results.keys())
        rmse_values = [lstm_results[m]['metrics']['rmse'] for m in methods]
        r2_values = [lstm_results[m]['metrics']['r2_score'] for m in methods]
        mae_values = [lstm_results[m]['metrics']['mae'] for m in methods]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # RMSE 比较
        ax1.bar(methods, rmse_values, color='skyblue', alpha=0.7)
        ax1.set_title('LSTM Model RMSE Comparison')
        ax1.set_ylabel('RMSE')
        ax1.set_xlabel('Alignment Method')
        for i, v in enumerate(rmse_values):
            ax1.text(i, v + max(rmse_values)*0.01, f'{v:.4f}', ha='center', va='bottom')
        
        # R² Score 比较
        ax2.bar(methods, r2_values, color='lightcoral', alpha=0.7)
        ax2.set_title('LSTM Model R² Score Comparison')
        ax2.set_ylabel('R² Score')
        ax2.set_xlabel('Alignment Method')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(r2_values):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        # MAE 比较
        ax3.bar(methods, mae_values, color='lightgreen', alpha=0.7)
        ax3.set_title('LSTM Model MAE Comparison')
        ax3.set_ylabel('MAE')
        ax3.set_xlabel('Alignment Method')
        for i, v in enumerate(mae_values):
            ax3.text(i, v + max(mae_values)*0.01, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('lstm_model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"LSTM性能比较图已保存为 lstm_model_performance_comparison.png")
        
        # --- 5. 测试最佳LSTM模型 ---
        if best_method and len(training_samples) > 0:
            print(f"\n--- 测试最佳LSTM模型 ({best_method}) ---")
            
            best_lstm_model = lstm_results[best_method]['model']
            # test_sample = training_samples[0]
            test_sample = 'chopped_M1-S0077.csv'
            
            results, test_metrics = test_lstm_model_on_sample(
                best_lstm_model, test_sample, best_method
            )
            
            if results and test_metrics:
                print(f"测试样本: {test_sample['filename_aw']}")
                print(f"测试 RMSE: {test_metrics['rmse']:.4f}")
                print(f"测试 R² Score: {test_metrics['r2_score']:.4f}")
                print(f"测试 MAE: {test_metrics['mae']:.4f}")
                print(f"预测序列长度: {test_metrics['sample_length']}")
                
                # 可视化预测结果
                true_values = results['true_values']
                predictions = results['predictions']
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                axis_labels = ['X', 'Y', 'Z', 'Magnitude']
                
                for i in range(3):
                    axes[i].plot(true_values[:, i], label=f'Real AW {axis_labels[i]}', alpha=0.8)
                    axes[i].plot(predictions[:, i], label=f'LSTM Predicted AW {axis_labels[i]}', 
                               linestyle='--', alpha=0.8)
                    axes[i].set_title(f'{axis_labels[i]} Axis - LSTM Prediction vs Real')
                    axes[i].set_xlabel('Time Step')
                    axes[i].set_ylabel('Acceleration')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                
                # 合成加速度对比
                true_mag = np.linalg.norm(true_values, axis=1)
                pred_mag = np.linalg.norm(predictions, axis=1)
                axes[3].plot(true_mag, label='Real Combined Acceleration', alpha=0.8)
                axes[3].plot(pred_mag, label='LSTM Predicted Combined Acceleration', 
                           linestyle='--', alpha=0.8)
                axes[3].set_title('Combined Acceleration - LSTM Prediction vs Real')
                axes[3].set_xlabel('Time Step')
                axes[3].set_ylabel('Acceleration')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'lstm_prediction_test_{best_method}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"LSTM预测结果可视化已保存为 lstm_prediction_test_{best_method}.png")
    
    print("\n=== LSTM映射模型训练完成 ===")
