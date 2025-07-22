"""
用mapping_model将Apple Watch数据映射为Axivity数据
"""

import joblib
import pandas as pd
import os
import numpy as np
import sys
from loguru import logger

# Configure loguru logger for transform module with colors
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

model_path = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/my_data/compare/best_mapping_model.pkl' if os.name != 'nt' else 'E:\\Raine\\OneDrive - Imperial College London\\IC\\70007 Individual Project\\Data\\my_data\\compare\\best_mapping_model.pkl'
mapping_model = joblib.load(model_path)

def load_data(dir):
    """
    load all csv files in the directory

    param: dir: directory containing csv files
    return: list that contains all dataframes
    """
    data = []
    files = []
    for file in os.listdir(dir):
        if file.endswith('.csv'):
            file_path = os.path.join(dir, file)
            df = pd.read_csv(file_path)
            data.append(df)
            files.append(file)
    return data, files

def get_dominant_frequency(fft_result, sampling_rate=20):
    """
    从FFT结果中获取主导频率
    
    参数:
        fft_result (array): FFT结果
        sampling_rate (float): 采样率(Hz)，默认为Apple Watch的20Hz
    
    返回:
        float: 主导频率(Hz)
    """
    # 计算FFT幅度
    magnitudes = np.abs(fft_result)
    
    # 只考虑前半部分（由于对称性）
    half_len = len(magnitudes) // 2
    
    # 找到最大幅度对应的索引
    peak_idx = np.argmax(magnitudes[:half_len])
    
    # 将索引转换为频率
    dominant_freq = peak_idx * (sampling_rate / len(fft_result))
    
    return dominant_freq

def extract_features(data):
    """
    提取与训练模型时相同的特征
    
    param: data: 包含X, Y, Z三轴加速度数据的DataFrame
    return: 字典，包含提取的特征
    """
    features = {}
    
    # 时域特征 - 计算均值、标准差、最大值、最小值
    mean_values = data.mean(axis=0)
    std_values = data.std(axis=0)
    max_values = data.max(axis=0)
    min_values = data.min(axis=0)
    
    for axis in ['X', 'Y', 'Z']:
        features[f'mean_{axis}'] = mean_values[axis]
        features[f'std_{axis}'] = std_values[axis]
        features[f'max_{axis}'] = max_values[axis]
        features[f'min_{axis}'] = min_values[axis]
    
    # 频域特征
    for axis in ['X', 'Y', 'Z']:
        fft_result = np.fft.fft(data[axis])
        features[f'dominant_freq_{axis}'] = get_dominant_frequency(fft_result)
    
    return features

def transform_data(df):
    """
    Transform Apple Watch data to Axivity data using the loaded model
    
    param: df: dataframe containing Apple Watch data
    return: dataframe with transformed data
    """
    try:
        # 检查模型期望的输入特征数量
        n_features_in = getattr(mapping_model, 'n_features_in_', None)
        # logger.debug(f"模型期望的特征数量: {n_features_in}")
        # logger.debug(f"输入数据的列: {df.columns}")
        
        # 假设输入数据包含加速度三轴数据
        if 'accelerationX' in df.columns and 'accelerationY' in df.columns and 'accelerationZ' in df.columns:
            # 预处理：重命名列以匹配训练时的格式
            data_copy = df.copy()
            data_copy['X'] = df['accelerationX']
            data_copy['Y'] = df['accelerationY']
            data_copy['Z'] = df['accelerationZ']
            
            # 提取特征
            apple_features = extract_features(data_copy[['X', 'Y', 'Z']])
            
            # 将特征转换为模型输入格式
            input_features = np.array(list(apple_features.values())).reshape(1, -1)
            
            # print(f"提取的特征数量: {input_features.shape[1]}")
            
            # 使用模型预测
            transformed_features = mapping_model.predict(input_features)
            
            # 预测结果是Axivity的特征值，我们需要把它们重新组合成时间序列
            # 这里我们假设我们只关心加速度数据的映射结果
            
            # 创建结果DataFrame - 保持与原始数据相同的长度
            result_df = pd.DataFrame(index=range(len(df)))
            
            # 将预测的Axivity特征均值应用到整个时间序列
            # 注意：这是一个简化处理，更复杂的处理可能需要考虑特征的时序变化
            mean_x = transformed_features[0][0]  # 假设第一个特征是X轴的均值
            mean_y = transformed_features[0][5]  # 假设第六个特征是Y轴的均值
            mean_z = transformed_features[0][10]  # 假设第十一个特征是Z轴的均值
            
            # 使用预测的均值和标准差重建时间序列
            std_x = transformed_features[0][1]  # X轴标准差
            std_y = transformed_features[0][6]  # Y轴标准差
            std_z = transformed_features[0][11]  # Z轴标准差
            
            # 简单地使用原始数据的形状和预测的统计特性创建新数据
            # 这里我们用标准化的原始数据乘以新的标准差并加上新的均值
            result_df['accelerationX'] = ((df['accelerationX'] - df['accelerationX'].mean()) / 
                                       (df['accelerationX'].std() + 1e-10)) * std_x + mean_x
            result_df['accelerationY'] = ((df['accelerationY'] - df['accelerationY'].mean()) / 
                                       (df['accelerationY'].std() + 1e-10)) * std_y + mean_y
            result_df['accelerationZ'] = ((df['accelerationZ'] - df['accelerationZ'].mean()) / 
                                       (df['accelerationZ'].std() + 1e-10)) * std_z + mean_z
            
            # 如果原始数据有时间戳列，则保留
            if 'Timestamp' in df.columns:
                result_df['Timestamp'] = df['Timestamp']
    
            return result_df
        else:
            logger.error(f"输入数据缺少必要的加速度列")
            return None
    except Exception as e:
        logger.error(f"转换数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载所有CSV文件
    dataframes, filenames = load_data(data_dir)
    
    # 对每个文件应用转换并保存结果
    for df, filename in zip(dataframes, filenames):
        logger.info(f"Processing {filename}...")
        transformed_df = transform_data(df)
        
        if transformed_df is not None:
            # 构建输出文件路径
            output_filename = f"mapped_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存转换后的数据
            transformed_df.to_csv(output_path, index=False)
            logger.info(f"Saved transformed data to {output_path}\n")

def map_from_custom_data(data:list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    从自定义数据目录加载数据并应用映射模型
    """
    transformed_data = []
    for df in data:
        transformed_df = transform_data(df)
        if transformed_df is not None:
            transformed_data.append(transformed_df)
    return transformed_data

if __name__ == "__main__":
    data_dir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/my_data'
    output_dir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/my_data/mapped_data'
    main()