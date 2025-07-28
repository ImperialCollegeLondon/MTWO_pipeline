"""
使用示例：配对训练集构建器

这个示例展示了如何使用修改后的 build_training_set.py 来创建配对的训练样本。
"""

import os
import pickle
import pandas as pd
import numpy as np

def load_paired_training_data(pickle_path: str):
    """
    加载配对的训练数据
    
    @param pickle_path: pickle文件路径
    @return: 训练样本列表
    """
    with open(pickle_path, 'rb') as f:
        training_samples = pickle.load(f)
    
    return training_samples

def demonstrate_usage():
    """
    演示如何使用配对的训练数据
    """
    # 假设已经运行了 build_training_set.py 并生成了配对数据
    pickle_path = "paired_training_data/paired_training_data.pkl"
    
    if not os.path.exists(pickle_path):
        print("请先运行 build_training_set.py 生成配对训练数据")
        return
    
    # 加载配对训练数据
    training_samples = load_paired_training_data(pickle_path)
    
    print(f"加载了 {len(training_samples)} 个配对训练样本")
    
    # 查看第一个样本的结构
    if training_samples:
        sample = training_samples[0]
        print(f"\n样本结构:")
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: numpy array, shape={value.shape}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
        
        # 展示如何使用配对数据进行训练
        print(f"\n使用示例:")
        print(f"AW 加速度数据形状: X={sample['aw_accelX'].shape}, Y={sample['aw_accelY'].shape}, Z={sample['aw_accelZ'].shape}")
        print(f"Vicon 加速度数据形状: X={sample['vicon_accelX'].shape}, Y={sample['vicon_accelY'].shape}, Z={sample['vicon_accelZ'].shape}")
        
        # 验证数据同步
        print(f"时间同步验证: AW和Vicon数据长度相同 = {len(sample['aw_accelX']) == len(sample['vicon_accelX'])}")

def create_training_matrix():
    """
    将配对数据转换为训练矩阵的示例
    """
    pickle_path = "paired_training_data/paired_training_data.pkl"
    
    if not os.path.exists(pickle_path):
        print("请先运行 build_training_set.py 生成配对训练数据")
        return
    
    training_samples = load_paired_training_data(pickle_path)
    
    # 创建输入特征矩阵 (AW数据) 和目标矩阵 (Vicon数据)
    all_aw_features = []
    all_vicon_targets = []
    
    for sample in training_samples:
        # AW数据作为输入特征 (N x 3)
        aw_features = np.column_stack([
            sample['aw_accelX'], 
            sample['aw_accelY'], 
            sample['aw_accelZ']
        ])
        
        # Vicon数据作为目标 (N x 3)
        vicon_targets = np.column_stack([
            sample['vicon_accelX'], 
            sample['vicon_accelY'], 
            sample['vicon_accelZ']
        ])
        
        all_aw_features.append(aw_features)
        all_vicon_targets.append(vicon_targets)
    
    # 合并所有样本
    X = np.vstack(all_aw_features)  # 输入特征矩阵
    y = np.vstack(all_vicon_targets)  # 目标矩阵
    
    print(f"训练特征矩阵 X 形状: {X.shape}")
    print(f"目标矩阵 y 形状: {y.shape}")
    print(f"总数据点数: {X.shape[0]}")
    
    return X, y

if __name__ == "__main__":
    print("=== 配对训练数据使用示例 ===")
    demonstrate_usage()
    
    print("\n=== 创建训练矩阵示例 ===")
    create_training_matrix()
