import os
import joblib
import pandas as pd
import datetime

from config import models, models_dir, save_dir
from trainAndEvaluation.train_and_evaluate import train_and_evaluate, plot_confusion_matrix

def train_and_evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test, getBest=False, mode='MTWO', 
                          class_names=None, save_confusion_matrices=False):
    # To store all model performance metrics
    model_results = {}
    
    # Select models based on mode
    selected_models = models.copy()
    if mode == 'MO':
        # For MO mode, use xgboost2 instead of xgboost
        if 'xgboost' in selected_models:
            del selected_models['xgboost']
        # Ensure xgboost2 is available for MO mode
        if 'xgboost2' not in selected_models:
            print("Warning: xgboost2 model not found in config for MO mode")
    elif mode == 'MTWO':
        # For MTWO mode, use xgboost instead of xgboost2
        if 'xgboost2' in selected_models:
            del selected_models['xgboost2']
        # Ensure xgboost is available for MTWO mode
        if 'xgboost' not in selected_models:
            print("Warning: xgboost model not found in config for MTWO mode")

    if getBest:
        # Track the best model based on AURC
        best_model_name = None
        best_accuracy = 0 
        best_aurc = float('inf')

    # Train and evaluate each model
    for model_name, model in selected_models.items():
        print(f"-------------------- [{model_name}] --------------------")
        accuracy, aurc, roc_auc, report, trained_model = train_and_evaluate(
            model, X_train_scaled, X_test_scaled, y_train, y_test, class_names=class_names
        )
        
        # Save the trained model
        model_filename = os.path.join(models_dir, f"{model_name}.pkl")
        joblib.dump(trained_model, model_filename)
        print(f"Model saved to {model_filename}")
        
        # Save confusion matrix if requested
        if save_confusion_matrices:
            # 创建cm目录
            cm_dir = os.path.join(save_dir, 'cm')
            os.makedirs(cm_dir, exist_ok=True)
            cm_save_path = os.path.join(cm_dir, f"confusion_matrix_{model_name}_{mode}.png")
            # Re-predict to get predictions for confusion matrix
            y_pred = trained_model.predict(X_test_scaled)
            # 修改plot_confusion_matrix函数调用，不显示图像
            from trainAndEvaluation.confusion_matrix_utils import save_confusion_matrix
            save_confusion_matrix(y_test, y_pred, class_names=class_names, 
                                model_name=model_name, save_path=cm_save_path)
        
        model_results[model_name] = {
            'accuracy': accuracy,
            'aurc': aurc,
            'roc_auc': roc_auc,
            'report': report
            }

        if getBest:
            # Store the best model based on AURC
            if aurc < best_aurc:
                best_aurc = aurc
                best_accuracy = accuracy
                best_model_name = model_name
                best_model = trained_model
        
        # Print results
        print(f"\n{model_name} Performance:")
        if accuracy is not None:
            print(f"Accuracy: {accuracy*100:.2f}%")
        if aurc is not None:
            print(f"AURC: {aurc:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Classification Report:\n{report}")

    if getBest:
        return model_results, (best_model_name, best_accuracy, best_aurc)
    else:
        return model_results, None
    
def save_comparison(model_results, best_model_info=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory '{save_dir}' created.")

    # Save model performance metrics to a CSV file
    metrics_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy': [model_results[m]['accuracy'] * 100 for m in model_results],
        'AURC': [model_results[m]['aurc'] for m in model_results],
        'ROC AUC': [model_results[m]['roc_auc'] if model_results[m]['roc_auc'] is not None else float('nan') 
                    for m in model_results]
    })
    metrics_df = metrics_df.sort_values('AURC')
    metrics_filename = os.path.join(save_dir, "model_comparison.csv")
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"Model comparison saved to {metrics_filename}")

    # Print the best model information if available
    if best_model_info:
        best_model_name, best_accuracy, best_aurc = best_model_info
    else:
        best_model_name = None
        best_accuracy = 0
        best_aurc = float('inf')
    print(f"\nBest model: '{best_model_name}' with AURC={best_aurc:.4f} and Accuracy: {best_accuracy*100:.2f}%")