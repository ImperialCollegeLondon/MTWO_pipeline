#!/usr/bin/env python3
"""
@ Author: Yufeng NA
@ Imperial College London  
@ Date: June 15, 2025
@ Description: 示例脚本 - 演示如何为已训练的模型绘制混淆矩阵
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split

# 导入配置和工具函数
from config import *
from trainAndEvaluation.confusion_matrix_utils import (
    plot_model_confusion_matrix, 
    plot_all_models_confusion_matrices, 
    compare_models_confusion_matrices
)

def load_test_data(mode='MTWO'):
    """
    加载测试数据
    这里你需要根据实际情况修改数据加载逻辑
    """
    print(f"Loading test data for {mode} mode...")
    
    # 这里应该加载你的实际测试数据
    # 示例代码 - 你需要替换为实际的数据加载逻辑
    try:
        # 尝试加载保存的测试数据
        X_test = np.load(os.path.join(cache_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(cache_dir, 'y_test.npy'))
        print(f"Test data loaded: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_test, y_test
    except FileNotFoundError:
        print("Cached test data not found. Please run the main training script first.")
        return None, None

def demo_single_model_confusion_matrix():
    """演示：绘制单个模型的混淆矩阵"""
    print("\n" + "="*60)
    print("演示1: 绘制单个模型的混淆矩阵")
    print("="*60)
    
    # 加载测试数据
    X_test, y_test = load_test_data(mode='MTWO')
    if X_test is None:
        print("无法加载测试数据，请先运行训练脚本")
        return
    
    # 设置类别名称
    class_names = ['Movement', 'Transport', 'Walking', 'Others']
    
    # 绘制XGBoost模型的混淆矩阵
    print("绘制XGBoost模型的混淆矩阵...")
    try:
        cm, save_path = plot_model_confusion_matrix(
            model_name='xgboost',
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            normalize=False,
            mode='MTWO'
        )
        print(f"✓ 成功绘制XGBoost混淆矩阵: {save_path}")
        
        # 也绘制标准化版本
        print("绘制标准化混淆矩阵...")
        cm_norm, save_path_norm = plot_model_confusion_matrix(
            model_name='xgboost',
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            normalize=True,
            mode='MTWO'
        )
        print(f"✓ 成功绘制标准化混淆矩阵: {save_path_norm}")
        
    except Exception as e:
        print(f"✗ 绘制失败: {e}")

def demo_all_models_confusion_matrices():
    """演示：绘制所有模型的混淆矩阵"""
    print("\n" + "="*60)
    print("演示2: 绘制所有可用模型的混淆矩阵")
    print("="*60)
    
    # 加载测试数据
    X_test, y_test = load_test_data(mode='MTWO')
    if X_test is None:
        print("无法加载测试数据，请先运行训练脚本")
        return
    
    # 设置类别名称
    class_names = ['Movement', 'Transport', 'Walking', 'Others']
    
    # 绘制所有模型的混淆矩阵
    print("绘制所有可用模型的混淆矩阵...")
    try:
        results = plot_all_models_confusion_matrices(
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            normalize=True,
            mode='MTWO'
        )
        print(f"✓ 成功为 {len(results)} 个模型绘制了混淆矩阵")
        
    except Exception as e:
        print(f"✗ 绘制失败: {e}")

def demo_model_comparison():
    """演示：比较多个模型的混淆矩阵"""
    print("\n" + "="*60)
    print("演示3: 比较多个模型的混淆矩阵")
    print("="*60)
    
    # 加载测试数据
    X_test, y_test = load_test_data(mode='MTWO')
    if X_test is None:
        print("无法加载测试数据，请先运行训练脚本")
        return
    
    # 设置类别名称和要比较的模型
    class_names = ['Movement', 'Transport', 'Walking', 'Others']
    models_to_compare = ['xgboost', 'rf', 'mlp']
    
    # 绘制模型比较图
    print(f"比较模型: {models_to_compare}")
    try:
        save_path = compare_models_confusion_matrices(
            X_test=X_test,
            y_test=y_test,
            models_to_compare=models_to_compare,
            class_names=class_names,
            mode='MTWO'
        )
        print(f"✓ 成功生成模型比较图: {save_path}")
        
    except Exception as e:
        print(f"✗ 比较失败: {e}")

def demo_mo_mode():
    """演示：MO模式下的混淆矩阵绘制"""
    print("\n" + "="*60)
    print("演示4: MO模式下的混淆矩阵绘制")
    print("="*60)
    
    # 加载测试数据 (MO模式)
    X_test, y_test = load_test_data(mode='MO')
    if X_test is None:
        print("无法加载MO模式测试数据")
        return
    
    # 设置类别名称
    class_names = ['Others', 'Movement']
    
    # 绘制XGBoost2模型的混淆矩阵（用于MO二分类）
    print("绘制MO模式下的混淆矩阵...")
    try:
        cm, save_path = plot_model_confusion_matrix(
            model_name='xgboost2',  # MO模式使用xgboost2
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            normalize=True,
            mode='MO'
        )
        print(f"✓ 成功绘制MO模式混淆矩阵: {save_path}")
        
    except Exception as e:
        print(f"✗ MO模式绘制失败: {e}")

def main():
    """主函数"""
    print("混淆矩阵绘制功能演示")
    print("确保您已经运行过训练脚本，以便有已训练的模型可用")
    
    # 检查模型目录
    if not os.path.exists(models_dir):
        print(f"模型目录不存在: {models_dir}")
        print("请先运行 main.py 进行模型训练")
        return
    
    # 列出可用的模型
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    if not model_files:
        print("没有找到已训练的模型文件")
        print("请先运行 main.py 进行模型训练")
        return
    
    print(f"找到 {len(model_files)} 个已训练的模型:")
    for f in model_files:
        print(f"  - {f}")
    
    # 运行演示
    demo_single_model_confusion_matrix()
    demo_all_models_confusion_matrices()
    demo_model_comparison()
    # demo_mo_mode()  # 如果有MO模式的模型，取消此行注释
    
    print("\n" + "="*60)
    print("演示完成！")
    print("所有混淆矩阵图片已保存到:", save_dir)
    print("="*60)

if __name__ == "__main__":
    main()
