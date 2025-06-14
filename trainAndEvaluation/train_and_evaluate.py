import numpy as np
import math
import gc
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from trainAndEvaluation.compute_aurc import compute_aurc

# Train and evaluate function in batches
def train_and_evaluate(model, X_train, X_test, y_train, y_test, batch_size=1000):
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
    
    return accuracy, aurc, roc_auc, report, model