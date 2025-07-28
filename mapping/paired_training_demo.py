"""
配对训练数据的映射模型使用示例

这个脚本展示了如何使用修改后的 mapping.py 来：
1. 加载配对训练数据
2. 应用不同的对齐方法
3. 训练映射模型
4. 评估和比较模型性能
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, r"E:\Raine\OneDrive - Imperial College London\IC\70007 Individual Project\MTWO_pipeline")

def demonstrate_paired_training():
    """
    演示配对训练数据的完整训练流程
    """
    print("=== 配对训练数据映射模型示例 ===\n")
    
    # 1. 检查配对训练数据是否存在
    paired_data_path = "paired_training_data/paired_training_data.pkl"
    
    if not os.path.exists(paired_data_path):
        print(f"错误: 找不到配对训练数据文件 {paired_data_path}")
        print("请先运行以下命令生成配对训练数据:")
        print("python build_training_set.py")
        return
    
    # 2. 模拟加载配对数据的过程
    print("1. 加载配对训练数据...")
    try:
        with open(paired_data_path, 'rb') as f:
            training_samples = pickle.load(f)
        
        print(f"   成功加载 {len(training_samples)} 个配对样本")
        
        # 显示样本统计信息
        total_duration = sum(sample['duration'] for sample in training_samples)
        total_points = sum(sample['length'] for sample in training_samples)
        avg_length = np.mean([sample['length'] for sample in training_samples])
        
        print(f"   总时长: {total_duration:.1f} 秒")
        print(f"   总数据点: {total_points} 个")
        print(f"   平均样本长度: {avg_length:.1f} 个数据点")
        
    except Exception as e:
        print(f"   加载失败: {str(e)}")
        return
    
    # 3. 展示不同对齐方法的概念
    print(f"\n2. 可用的对齐方法:")
    alignment_methods = {
        'rotation_matrix': '基于重力向量计算旋转矩阵',
        'procrustes': '使用Procrustes分析进行坐标系对齐',
        'none': '不进行坐标系对齐'
    }
    
    for method, description in alignment_methods.items():
        print(f"   - {method}: {description}")
    
    # 4. 展示训练流程概念
    print(f"\n3. 训练流程:")
    print("   a) 加载配对样本")
    print("   b) 应用坐标系对齐")
    print("   c) 合并所有对齐后的数据")
    print("   d) 训练线性回归模型 (Vicon -> Apple Watch)")
    print("   e) 评估模型性能 (RMSE, R²)")
    
    # 5. 展示期望的输出文件结构
    print(f"\n4. 训练完成后的文件结构:")
    for method in alignment_methods.keys():
        print(f"   mapping_models_{method}/")
        print(f"   ├── mapping_model.joblib        (训练好的模型)")
        print(f"   ├── alignment_info.pkl         (对齐信息)")
        print(f"   └── training_metrics.pkl       (训练指标)")
        print()
    
    # 6. 展示如何使用训练好的模型
    print("5. 模型使用示例:")
    print("   # 加载最佳模型")
    print("   model = joblib.load('mapping_models_rotation_matrix/mapping_model.joblib')")
    print("   ")
    print("   # 准备Vicon输入数据 (N x 3: accelX, accelY, accelZ)")
    print("   vicon_input = your_vicon_data[['accelX', 'accelY', 'accelZ']].values")
    print("   ")
    print("   # 预测Apple Watch数据")
    print("   aw_predicted = model.predict(vicon_input)")
    print("   ")
    print("   # aw_predicted 将包含预测的Apple Watch三轴加速度数据")
    
    # 7. 如果存在训练结果，展示性能对比
    print(f"\n6. 检查已有的训练结果:")
    
    results_found = False
    for method in alignment_methods.keys():
        metrics_path = f"mapping_models_{method}/training_metrics.pkl"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
                
                if not results_found:
                    print("   找到以下训练结果:")
                    print(f"   {'方法':<15} {'RMSE':<10} {'R² Score':<10} {'样本数':<10}")
                    print("   " + "-" * 50)
                    results_found = True
                
                print(f"   {method:<15} {metrics['rmse']:<10.4f} {metrics['r2_score']:<10.4f} {metrics['n_samples']:<10}")
                
            except Exception as e:
                print(f"   {method}: 无法读取训练结果 ({str(e)})")
    
    if not results_found:
        print("   未找到训练结果，请运行 python mapping.py 进行训练")
    
    print(f"\n=== 示例完成 ===")
    print("要开始实际训练，请运行: python mapping.py")


def create_sample_config():
    """
    创建一个示例配置文件
    """
    config = {
        'paired_data_path': 'paired_training_data/paired_training_data.pkl',
        'alignment_methods': ['rotation_matrix', 'procrustes', 'none'],
        'output_dir_prefix': 'mapping_models',
        'visualization': {
            'save_plots': True,
            'show_plots': False,
            'plot_dpi': 300
        },
        'model_settings': {
            'model_type': 'LinearRegression',
            'normalize_features': False,
            'test_split': 0.2
        }
    }
    
    config_path = 'mapping_config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"示例配置文件已创建: {config_path}")
    return config_path


if __name__ == "__main__":
    # 运行演示
    demonstrate_paired_training()
    
    # 创建示例配置
    print(f"\n" + "="*60)
    create_sample_config()
