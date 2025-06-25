"""
@ Author: Yufeng NA
@ Imperial College London
@ Date: June 15, 2025
@ Description: Utility functions for plotting confusion matrices of trained models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from sklearn.metrics import confusion_matrix

# Add the parent directory to the Python path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import models_dir, save_dir

def plot_model_confusion_matrix(model_name, X_test, y_test, class_names=None, 
                               save_path=None, normalize=False, mode='MTWO'):
    """
    加载训练好的模型并绘制其在测试集上的混淆矩阵
    
    参数:
    - model_name: 模型名称 (如 'xgboost', 'rf', 'mlp' 等)
    - X_test: 测试特征
    - y_test: 测试标签
    - class_names: 类别名称列表
    - save_path: 保存路径，如果为None则自动生成
    - normalize: 是否标准化混淆矩阵
    - mode: 模式 ('MTWO' 或 'MO')
    """
    # 加载模型
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    # 进行预测
    y_pred = model.predict(X_test)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 设置默认类别名称
    if class_names is None:
        unique_labels = sorted(set(list(y_test) + list(y_pred)))
        if mode == 'MO':
            class_names = ['Others', 'Movement'] if len(unique_labels) == 2 else [f'Class_{i}' for i in unique_labels]
        else:  # MTWO
            default_names = {0: 'Movement', 1: 'Transport', 2: 'Walking', 3: 'Others'}
            class_names = [default_names.get(i, f'Class_{i}') for i in unique_labels]
    
    # 设置保存路径
    if save_path is None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = '_normalized' if normalize else ''
        save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}_{mode}{suffix}.png")
    
    # 绘制混淆矩阵
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f'Normalized Confusion Matrix - {model_name.upper()} ({mode})'
        plot_cm = cm_norm
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {model_name.upper()} ({mode})'
        plot_cm = cm
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(plot_cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                square=True, linewidths=0.5)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    
    # 添加性能指标
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, save_path

def plot_all_models_confusion_matrices(X_test, y_test, class_names=None, 
                                     normalize=False, mode='MTWO', 
                                     models_to_plot=None):
    """
    为所有训练好的模型绘制混淆矩阵
    
    参数:
    - X_test: 测试特征
    - y_test: 测试标签
    - class_names: 类别名称列表
    - normalize: 是否标准化混淆矩阵
    - mode: 模式 ('MTWO' 或 'MO')
    - models_to_plot: 要绘制的模型列表，如果为None则绘制所有可用模型
    """
    # 获取所有可用的模型文件
    if models_to_plot is None:
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        models_to_plot = [f.replace('.pkl', '') for f in model_files]
    
    print(f"Plotting confusion matrices for models: {models_to_plot}")
    
    results = {}
    for model_name in models_to_plot:
        try:
            cm, save_path = plot_model_confusion_matrix(
                model_name, X_test, y_test, class_names=class_names,
                normalize=normalize, mode=mode
            )
            results[model_name] = {'confusion_matrix': cm, 'save_path': save_path}
            print(f"✓ Successfully plotted confusion matrix for {model_name}")
        except Exception as e:
            print(f"✗ Failed to plot confusion matrix for {model_name}: {e}")
    
    return results

def compare_models_confusion_matrices(X_test, y_test, models_to_compare, 
                                    class_names=None, mode='MTWO'):
    """
    在一个图中比较多个模型的混淆矩阵
    
    参数:
    - X_test: 测试特征
    - y_test: 测试标签
    - models_to_compare: 要比较的模型名称列表
    - class_names: 类别名称列表
    - mode: 模式 ('MTWO' 或 'MO')
    """
    n_models = len(models_to_compare)
    if n_models == 0:
        print("No models to compare")
        return
    
    # 创建子图
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    # 设置默认类别名称
    if class_names is None:
        unique_labels = sorted(set(y_test))
        if mode == 'MO':
            class_names = ['Others', 'Movement'] if len(unique_labels) == 2 else [f'Class_{i}' for i in unique_labels]
        else:  # MTWO
            default_names = {0: 'Movement', 1: 'Transport', 2: 'Walking', 3: 'Others'}
            class_names = [default_names.get(i, f'Class_{i}') for i in unique_labels]
    
    # 收集所有混淆矩阵数据，用于统一的colorbar
    all_cms = []
    
    for i, model_name in enumerate(models_to_compare):
        try:
            # 加载模型
            model_path = os.path.join(models_dir, f"{model_name}.pkl")
            model = joblib.load(model_path)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            all_cms.append((cm, cm_norm))
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            all_cms.append((None, None))
    
    # 计算全局的colorbar范围
    valid_cms = [cm_norm for cm, cm_norm in all_cms if cm_norm is not None]
    if valid_cms:
        vmin = min(cm.min() for cm in valid_cms)
        vmax = max(cm.max() for cm in valid_cms)
    else:
        vmin, vmax = 0, 1
    
    # 绘制每个模型的混淆矩阵
    for i, (model_name, (cm, cm_norm)) in enumerate(zip(models_to_compare, all_cms)):
        if cm_norm is not None:
            # 只在最后一个子图上显示colorbar
            show_cbar = (i == len(models_to_compare) - 1)
            
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[i], square=True, cbar=show_cbar,
                       vmin=vmin, vmax=vmax,
                       cbar_kws={'label': 'Proportion'} if show_cbar else None)
            
            axes[i].set_title(f'{model_name.upper()}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Predicted', fontsize=12)
            if i == 0:
                axes[i].set_ylabel('True', fontsize=12)
            
            # 添加准确率
            accuracy = np.trace(cm) / np.sum(cm)
            axes[i].text(0.5, -0.15, f'Acc: {accuracy:.3f}', 
                        transform=axes[i].transAxes, ha='center', fontsize=11)
            
        else:
            axes[i].text(0.5, 0.5, f'Error loading\n{model_name}', 
                        transform=axes[i].transAxes, ha='center', va='center')
    
    plt.suptitle(f'Model Comparison - Confusion Matrices ({mode} Mode)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存比较图到cm目录
    cm_dir = os.path.join(save_dir, 'cm')
    os.makedirs(cm_dir, exist_ok=True)
    save_path = os.path.join(cm_dir, f"confusion_matrices_comparison_{mode}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison saved to: {save_path}")
    
    plt.close()  # 不显示，只保存
    
    return save_path

def save_confusion_matrix(y_true, y_pred, class_names=None, model_name="Model", save_path=None, normalize=False):
    """
    只保存混淆矩阵，不显示
    
    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - class_names: 类别名称列表，如果为None则使用数字标签
    - model_name: 模型名称，用于图表标题
    - save_path: 保存路径
    - normalize: 是否标准化混淆矩阵
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f'Normalized Confusion Matrix - {model_name}'
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {model_name}'
    
    # 设置图形大小
    plt.figure(figsize=(8, 6))
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # 只保存，不显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()  # 关闭图形，释放内存
    
    return cm

# 使用示例函数
def example_usage():
    """
    使用示例
    """
    print("="*60)
    print("混淆矩阵绘制功能使用示例")
    print("="*60)
    print()
    print("1. 绘制单个模型的混淆矩阵:")
    print("   from trainAndEvaluation.confusion_matrix_utils import plot_model_confusion_matrix")
    print("   plot_model_confusion_matrix('xgboost', X_test, y_test, class_names=['Movement', 'Transport', 'Walking', 'Others'])")
    print()
    print("2. 绘制所有模型的混淆矩阵:")
    print("   from trainAndEvaluation.confusion_matrix_utils import plot_all_models_confusion_matrices")
    print("   plot_all_models_confusion_matrices(X_test, y_test, normalize=True)")
    print()
    print("3. 比较多个模型的混淆矩阵:")
    print("   from trainAndEvaluation.confusion_matrix_utils import compare_models_confusion_matrices")
    print("   compare_models_confusion_matrices(X_test, y_test, ['xgboost', 'rf', 'mlp'])")
    print()
    print("="*60)

if __name__ == "__main__":
    example_usage()
