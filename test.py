'''
@ Author: Yufeng NA
@ Imperial College London
@ Date: June 5, 2025
@ Description: Test script to perform activity recognition on given data sample(s).
# ------------------Test Script-----------------------------
# This script is designed to test the machine learning pipeline for activity recognition on a given set of data samples.
'''
import pandas as pd
import numpy as np
import os
import joblib
import sys
from tqdm import tqdm
from collections import Counter
import datetime
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add the current directory to the Python path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featureExtractor.features import compute_features, compute_features_MO

from config import getLogger
# Initialize logger
logger = getLogger('info')

import testInit.init
import testInit.test_config
from config import rootDir, encode_path, scaler_path, pca_model_path
from config import WINDOW_SIZE



# ============================
# Following is the core code
# ============================



# --------load test data from csv file-----------------
def predict(model, data_path, mode='MTWO'):
    '''The prediction pipeline for a single data file'''
    from dataLoader.load_data import parse_csv
    df = parse_csv(data_path, useFilter=True)  # Load (and filter) the data from CSV file

    # 2. sliding window
    from dataTransformer.sliding_window import splitIntoOverlappingWindows
    windows = splitIntoOverlappingWindows(df)

    # 3. feature extraction
    features_list = []
    for window_df in windows:
        if len(window_df) < WINDOW_SIZE:
            continue
        # Choose feature extraction function based on mode
        if mode == 'MO':
            features_list.append(compute_features_MO(window_df))
        else:  # MTWO mode
            features_list.append(compute_features(window_df))
    X_features = np.array(features_list)

    logger.debug(f"Features before scaling: {X_features.shape}")
    
    # 4. Scale the features
    scaler = joblib.load(scaler_path)
    X_features = scaler.transform(X_features)
    logger.debug(f"Features after scaling: {X_features.shape}")

    # 5. PCA 
    pca_model = joblib.load(pca_model_path)
    X_features = pca_model.transform(X_features)
    logger.debug(f"Features after PCA: {X_features.shape}")
    
    # 检查模型期望的特征数量
    expected_features = getattr(model, 'n_features_in_', 'Unknown')
    logger.debug(f"Model expects {expected_features} features, providing {X_features.shape[1]} features")

    # 6. Predict
    predictions = model.predict_proba(X_features)
    predicted_classes = np.argmax(predictions, axis=1)
    encoder = joblib.load(encode_path)
    predicted_labels = encoder.inverse_transform(predicted_classes)

    # 7. Convert to DataFrame and Save
    results = pd.DataFrame({
        'prediction': predicted_labels,
        'prob': predictions.max(axis=1)
    })
    
    return results

def predict_all(file_list, data_dir, mode='MTWO'):
    model_dics = testInit.test_config.model_dics_mo if mode == 'MO' else testInit.test_config.model_dics
    
    results = {
        'overall': {'tp': Counter(), 'fp': Counter(), 'fn': Counter(), 'total': Counter()},
        'movement': {'tp': Counter(), 'fp': Counter(), 'fn': Counter(), 'total': Counter()},
        'transport': {'tp': Counter(), 'fp': Counter(), 'fn': Counter(), 'total': Counter()},
        'walking': {'tp': Counter(), 'fp': Counter(), 'fn': Counter(), 'total': Counter()},
        'others': {'tp': Counter(), 'fp': Counter(), 'fn': Counter(), 'total': Counter()}
    } if mode == 'MTWO' else {
        'overall': {'tp': Counter(), 'fp': Counter(), 'fn': Counter(), 'total': Counter()},
        'movement': {'tp': Counter(), 'fp': Counter(), 'fn': Counter(), 'total': Counter()},
        'others': {'tp': Counter(), 'fp': Counter(), 'fn': Counter(), 'total': Counter()}
    }

    # 添加混淆矩阵存储
    confusion_matrices = {}

    model_name_list = list(model_dics.keys())
    # model_name_list = ['xgboost']
    for model_name in tqdm(model_name_list, desc="Evaluation Progress", leave=False):
        Error_file_cnt = 0
        
        # 为每个模型收集所有预测和真实标签
        all_predictions = []
        all_ground_truths = []
        
        for file_name in tqdm(file_list, desc=f"Current model = {model_name}", leave=False, colour="green"):
            file_basename = os.path.join(os.path.splitext(file_name)[0], model_name)
            # try:
            model_file = joblib.load(model_dics[model_name])
            file_path = os.path.join(data_dir, file_name)  # Create full file path
            prediction_results = predict(model_file, file_path, mode=mode)

            if prediction_results is None:
                logger.warning(f"[Warning] {file_name} too short, skipping.")
                continue
            
            # ATTENTION!!!!!!!!!!!!!!!!!!!!
            # ATTENTION!!!!!!!!!!!!!!!!!!!!
            # ATTENTION!!!!!!!!!!!!!!!!!!!!
            ground_truth_dic = testInit.test_config.ground_truth_dic_mo if mode == 'MO' else testInit.test_config.ground_truth_dic
            ground_truth = ground_truth_dic.get(file_name.split("-")[3][-1], 'Unknown')
            # ground_truth = 'M'  # Test the new GERF data, all ground truth is 'M'

            preds = prediction_results['prediction'].values
            
            # 收集预测和真实标签用于混淆矩阵
            all_predictions.extend(preds)
            all_ground_truths.extend([ground_truth] * len(preds))
            
            # Calculate metrics for overall
            tp_overall = np.sum(preds == ground_truth)
            fp_overall = np.sum(preds != ground_truth)
            fn_overall = 0  # For overall, we consider all samples
            
            results['overall']['tp'][model_name] += tp_overall
            results['overall']['fp'][model_name] += fp_overall
            results['overall']['total'][model_name] += len(preds)

            # # DEBUG
            logger.debug(f"Ground truth: {ground_truth}")
            logger.debug(f"Predictions: {preds}")
            
            # Update category-specific results
            category_map = {'M': 'movement', 'T': 'transport', 'W': 'walking', 'O': 'others'} if mode == 'MTWO' else {'M': 'movement', 'O': 'others'}
            
            # For each category, calculate TP, FP, FN
            for gt_label, category in category_map.items():
                tp = np.sum((preds == gt_label) & (ground_truth == gt_label))
                fp = np.sum((preds == gt_label) & (ground_truth != gt_label))
                fn = np.sum((preds != gt_label) & (ground_truth == gt_label))
                
                results[category]['tp'][model_name] += tp
                results[category]['fp'][model_name] += fp
                results[category]['fn'][model_name] += fn
                
                # Total samples for this category
                if ground_truth == gt_label:
                    results[category]['total'][model_name] += len(preds)
            
            # except Exception as e:
            #     logger.error(f"{file_basename} with model {model_name} failed: {e}")
            #     Error_file_cnt += 1
            #     continue
        # 计算该模型的混淆矩阵
        if all_predictions and all_ground_truths:
            confusion_matrices[model_name] = {
                'predictions': all_predictions,
                'ground_truths': all_ground_truths
            }
        
        # print(f"{model_name} completed with {Error_file_cnt} errors.")
    
    # 将混淆矩阵添加到结果中
    results['confusion_matrices'] = confusion_matrices
    return results

def calculate_metrics(tp, fp, fn, total):
    """Calculate accuracy, precision, recall, and F1 score"""
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    accuracy = tp / total * 100 if total > 0 else 0.0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1

def display_accuracy(results, mode='MTWO'):
    model_dics = testInit.test_config.model_dics if mode == 'MTWO' else testInit.test_config.model_dics_mo
    model_name_list = list(model_dics.keys())
    class_list = ['movement', 'transport', 'walking', 'others'] if mode == 'MTWO' else ['movement', 'others']

    print('='*80)
    print("Performance Metrics on Real World Data (Per Window)")
    print('='*80)

    # Display overall metrics
    print("\nOverall Performance:")
    display_category_metrics_table(results['overall'], model_name_list, is_overall=True)

    # Display metrics for each category
    for category_name in class_list:
        print(f"\n {category_name.capitalize()} Performance:")
        display_category_metrics_table(results[category_name], model_name_list, is_overall=False)

    print('='*80)

def display_category_metrics_table(category_results, model_name_list, is_overall=False):
    """Display metrics in a beautiful table format"""
    table_data = []
    headers = ["Model", "Accuracy (%)", "Correct/Total"] if is_overall else ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "Correct/Total"]
    
    for model_name in model_name_list:
        tp = category_results['tp'][model_name]
        fp = category_results['fp'][model_name]
        fn = category_results['fn'][model_name]
        total = category_results['total'][model_name]
        
        if total > 0:
            if is_overall:
                # For overall metrics, use accuracy calculation
                accuracy = tp / total * 100
                row = [
                    model_name.capitalize(),
                    f"{accuracy:.2f}",
                    f"{tp}/{total}"
                ]
            else:
                # For category-specific metrics, calculate all metrics
                accuracy, precision, recall, f1 = calculate_metrics(tp, fp, fn, total)
                row = [
                    model_name.capitalize(),
                    f"{accuracy:.2f}",
                    f"{precision:.2f}",
                    f"{recall:.2f}",
                    f"{f1:.2f}",
                    f"{tp}/{total}"
                ]
            table_data.append(row)
        else:
            # Handle case where no samples exist for this category
            if is_overall:
                row = [model_name.capitalize(), "N/A", "0/0"]
            else:
                row = [model_name.capitalize(), "N/A", "N/A", "N/A", "N/A", "0/0"]
            table_data.append(row)
            
    if table_data:
        table = tabulate(table_data, headers=headers, tablefmt="grid", numalign="center", stralign="center")
        # Split table into lines and log each line
        for line in table.split('\n'):
            print(line)
        
        # Check for any models with no samples and log warnings
        for i, model_name in enumerate(model_name_list):
            total = category_results['total'][model_name]
            if total == 0:
                logger.warning(f"{model_name}: No samples belong to this category! Please check the pattern settings or file path of the ground truth.")
    else:
        logger.warning("No data to display for this category!")

def visualize_results(results, mode='MTWO'):
    """Visualize the results of the predictions"""
    import matplotlib.pyplot as plt

    # Get model names from the Counter objects
    model_dics = testInit.test_config.model_dics if mode == 'MTWO' else testInit.test_config.model_dics_mo
    model_names = list(model_dics.keys())
    
    # Extract overall metrics for each model
    overall_results = results['overall']
    tp_values = [overall_results['tp'][model] for model in model_names]
    fp_values = [overall_results['fp'][model] for model in model_names]
    fn_values = [overall_results['fn'][model] for model in model_names]
    total_values = [overall_results['total'][model] for model in model_names]
    
    accuracy_values = [tp / total * 100 if total > 0 else 0 for tp, total in zip(tp_values, total_values)]
    precision_values = [tp / (tp + fp) * 100 if (tp + fp) > 0 else 0 for tp, fp in zip(tp_values, fp_values)]
    recall_values = [tp / (tp + fn) * 100 if (tp + fn) > 0 else 0 for tp, fn in zip(tp_values, fn_values)]
    f1_values = [2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 for precision, recall in zip(precision_values, recall_values)]
    
    # Create a DataFrame for visualization
    import pandas as pd
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values
    })
    # Set the model names as index
    metrics_df.set_index('Model', inplace=True)
    # Plotting
    plt.figure(figsize=(12, 8))
    metrics_df.plot(kind='bar', rot=0)
    plt.title('Model Performance Metrics')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig(rf"test_results.png")

def save_confusion_matrices(results, mode='MTWO'):
    """计算、显示并保存混淆矩阵"""
    if 'confusion_matrices' not in results:
        logger.warning("No confusion matrix data found in results")
        return
    
    # 确保保存目录存在
    save_dir = os.path.join(rootDir, "result_compare")
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义类别标签
    if mode == 'MTWO':
        labels = ['M', 'T', 'W', 'O']
        label_names = ['Movement', 'Transport', 'Walking', 'Others']
    else:
        labels = ['M', 'O'] 
        label_names = ['Movement', 'Others']
    
    confusion_data = results['confusion_matrices']
    
    # 为每个模型生成混淆矩阵
    for model_name, data in confusion_data.items():
        predictions = data['predictions']
        ground_truths = data['ground_truths']
        
        # 计算混淆矩阵
        cm = confusion_matrix(ground_truths, predictions, labels=labels)
        
        # 创建可视化
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f'Confusion Matrix - {model_name.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 保存图像
        from time import strftime, localtime
        timestamp = strftime('%Y%m%d_%H%M%S', localtime())
        save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印混淆矩阵到控制台
        print(f"\nConfusion Matrix for {model_name.capitalize()}:")
        print("=" * 50)
        
        # 创建表格形式的混淆矩阵
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        print(tabulate(cm_df, headers=label_names, tablefmt="grid", numalign="center"))
        
        # 保存混淆矩阵数据到CSV
        csv_path = os.path.join(save_dir, f"confusion_matrix_{model_name}_{timestamp}.csv")
        cm_df.to_csv(csv_path)
        
        logger.info(f"Confusion matrix for {model_name} saved to {save_path} and {csv_path}")

if __name__ == '__main__':

    # mode = 'MO'
    mode = 'MTWO'

    data_dir = os.path.join(rootDir, r"Data/my_data/") # Without mapping model (apple watch)
    # data_dir = os.path.join(rootDir, r"Data/my_data/mapped_data") # With mapping model (converted to AX form)

    # 1. Load all csv name from the directory
    # file_list contains all file basenames in the data_dir user provided
    # pattern_style = 'gerf'
    pattern_style = 'gerf_mo' if mode == 'MO' else 'gerf'
    file_list = testInit.init.get_gerf_files(data_dir, pattern_style=pattern_style)


    # 2. Initialise the results CSV file
    # testInit.init.init_res_csv(data_dir)


    # 3. Predict each file for each model
    results = predict_all(file_list, data_dir, mode=mode)


    # 4. Compute and display the accuracy
    display_accuracy(results, mode=mode)
    visualize_results(results, mode=mode)

    # 5. 保存混淆矩阵
    save_confusion_matrices(results, mode=mode)

    # 6. Save the results
    # testInit.init.save_res_csv(data_dir)
    # print(f"All predictions completed and saved to {data_dir}.")