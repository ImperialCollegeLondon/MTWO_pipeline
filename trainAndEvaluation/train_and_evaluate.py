import numpy as np
import math
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import os

from trainAndEvaluation.compute_aurc import compute_aurc

def plot_confusion_matrix(y_true, y_pred, class_names=None, model_name="Model", save_path=None, normalize=False):
    """
    绘制混淆矩阵
    
    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - class_names: 类别名称列表，如果为None则使用数字标签
    - model_name: 模型名称，用于图表标题
    - save_path: 保存路径，如果为None则只显示不保存
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
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm

# Train and evaluate function in batches
def train_and_evaluate(model, X_train, X_test, y_train, y_test, class_names=None, batch_size=1000):
    """Train a model and evaluate its performance with batch processing."""
    model.fit(X_train, y_train)

    n_batches = math.ceil(len(X_test) / batch_size) # Round up to put the remaining samples in the last batch
    y_pred_list = []
    y_confidence_list = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_test))
        X_batch = X_test[start_idx:end_idx]
        
        # 特殊处理XGBoost二分类
        if str(type(model).__name__) == 'XGBClassifier':
            # XGBoost分类器处理
            batch_pred = model.predict(X_batch)
        else:
            batch_pred = model.predict(X_batch)
        
        y_pred_list.append(batch_pred)
        
        # 获取置信度分数
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_batch)
            batch_confidence = np.max(proba, axis=1)
            y_confidence_list.append(batch_confidence)
        else:
            batch_decision = model.decision_function(X_batch) if hasattr(model, "decision_function") else np.ones_like(batch_pred)
            y_confidence_list.append(batch_decision)
    
    # Concatenate results from all batches
    y_pred = np.concatenate(y_pred_list)
    y_confidence = np.concatenate(y_confidence_list)
    
    # Free memory
    del y_pred_list, y_confidence_list
    gc.collect()
    
    accuracy = accuracy_score(y_test, y_pred)
    aurc = compute_aurc(y_test, y_pred, y_confidence)
    
    # 计算ROC AUC（对于概率模型）
    if hasattr(model, "predict_proba"):
        y_proba_list = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_test))
            y_proba_list.append(model.predict_proba(X_test[start_idx:end_idx]))
        y_proba = np.concatenate(y_proba_list)
        
        # 对于二分类，只使用正类的概率
        if y_proba.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        
        del y_proba_list, y_proba
        gc.collect()
    else:
        print(f"{model} does not support probability predictions.")
        roc_auc = None
    
    # 生成分类报告
    report = classification_report(y_test, y_pred)
    
    # 不在这里绘制混淆矩阵，避免重复绘制
    # 混淆矩阵将在 train_and_evaluate_all 中统一处理
    
    return accuracy, aurc, roc_auc, report, model