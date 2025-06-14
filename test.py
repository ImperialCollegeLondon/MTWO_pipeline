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
from tqdm import tqdm
from collections import Counter
import datetime

from featureExtractor.features import compute_features, compute_features_MO
import testInit.init
import testInit.test_config
from config import rootDir, encode_path, scaler_path, pca_model_path
from config import WINDOW_SIZE

# --------load test data from csv file-----------------
def predict(model, data_path, mode='MTWO'):
    '''The prediction pipeline for a single data file'''
    # print(f"[info@test] -> Starting prediction for {data_path} with mode {mode}...")
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
            features = compute_features_MO(window_df)
        else:  # MTWO mode
            features = compute_features(window_df)
        features_list.append(features)
    X_features = np.array(features_list)

    # 4. Scale the features
    scaler = joblib.load(scaler_path)
    X_features = scaler.transform(X_features)

    # 5. PCA 
    pca_model = joblib.load(pca_model_path)
    # print(f"[info@test] -> PCA model loaded from {pca_model_path}.")
    X_features = pca_model.transform(X_features)

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
        'overall': {'correct': Counter(), 'total': Counter()},
        'movement': {'correct': Counter(), 'total': Counter()},
        'transport': {'correct': Counter(), 'total': Counter()},
        'walking': {'correct': Counter(), 'total': Counter()},
        'others': {'correct': Counter(), 'total': Counter()}
    } if mode == 'MTWO' else {
        'overall': {'correct': Counter(), 'total': Counter()},
        'movement': {'correct': Counter(), 'total': Counter()},
        'others': {'correct': Counter(), 'total': Counter()}
    }

    model_name_list = list(model_dics.keys())
    # model_name_list = ['xgboost']
    for model_name in tqdm(model_name_list, desc="Evaluation Progress", leave=False):
        Error_file_cnt = 0
        for file_name in tqdm(file_list, desc=f"Current model = {model_name}", leave=False, colour="green"):
            file_basename = os.path.join(os.path.splitext(file_name)[0], model_name)
            # try:
            model_file = joblib.load(model_dics[model_name])
            file_path = os.path.join(data_dir, file_name)  # Create full file path
            prediction_results = predict(model_file, file_path, mode=mode)

            if prediction_results is None:
                print(f"[Warning] {file_name} too short, skipping.")
                continue
            
            # ATTENTION!!!!!!!!!!!!!!!!!!!!
            # ATTENTION!!!!!!!!!!!!!!!!!!!!
            # ATTENTION!!!!!!!!!!!!!!!!!!!!
            ground_truth = testInit.test_config.ground_truth_dic[file_name.split("-")[3][-1]]
            # ground_truth = 'M'  # Test the new GERF data, all ground truth is 'M'

            preds = prediction_results['prediction'].values
            correct = np.sum(preds == ground_truth)

            # # DEBUG
            # print(ground_truth)
            # print(preds)
            
            # Update overall results
            results['overall']['correct'][model_name] += correct
            results['overall']['total'][model_name] += len(preds)

            # Update category-specific results
            category_map = {'M': 'movement', 'T': 'transport', 'W': 'walking', 'O': 'others'} if mode == 'MTWO' else {'M': 'movement', 'O': 'others'}
            if ground_truth in category_map:
                category = category_map[ground_truth]
                results[category]['correct'][model_name] += correct
                results[category]['total'][model_name] += len(preds)
            # except Exception as e:
            #     print(f"[Error@test] {file_basename} with model {model_name} failed: {e}")
            #     Error_file_cnt += 1
            #     continue
        # print(f"[info@test] {model_name} completed with {Error_file_cnt} errors.")
    return results

def display_accuracy(results, mode='MTWO'):
    model_dics = testInit.test_config.model_dics if mode == 'MTWO' else testInit.test_config.model_dics_mo
    model_name_list = list(model_dics.keys())
    class_list = ['movement', 'transport', 'walking', 'others'] if mode == 'MTWO' else ['movement', 'others']
    
    print('\n------------------------------------------')
    print("Accuracies on real world data (per window):")

    # Display overall accuracy
    display_category_accuracy(results['overall'], model_name_list)

    # Display accuracy for each category
    for category_name in class_list:
        print(f"\n{category_name.capitalize()} accuracy (per window):")
        display_category_accuracy(results[category_name], model_name_list)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nLast test run at {current_time}")
    print('------------------------------------------')
            
def display_category_accuracy(category_results, model_name_list):
    """Display accuracy for a specific category"""
    for model_name in model_name_list:
        total = category_results['total'][model_name]
        if total > 0:
            acc = category_results['correct'][model_name] / total * 100
            print(f"{model_name}: {acc:.2f}%")
        else:
            print(f"The denominator for {model_name} is {total}, cannot compute accuracy.")

if __name__ == '__main__':

    mode = 'MO'

    # data_dir = os.path.join(rootDir, r"Data/my_data/") # Without mapping model (apple watch)
    data_dir = os.path.join(rootDir, r"Data/my_data/mapped_data") # With mapping model (converted to AX form)
    # data_dir = os.path.join(rootDir, r'Data/my_Movement_data') # The new gerf data (all movement)

    # 1. Load all csv name from the directory
    # file_list contains all file basenames in the data_dir user provided
    file_list = testInit.init.get_gerf_files(data_dir, pattern_style='gerf')

    # 2. Initialise the results CSV file
    testInit.init.init_res_csv(data_dir)

    # 3. Predict each file for each model
    results = predict_all(file_list, data_dir, mode=mode)  # 根据你的模型选择正确的模式

    # 4. Compute and display the accuracy
    display_accuracy(results, mode=mode)  # 根据你的模型选择正确的模式

    # 5. Save the results
    # testInit.init.save_res_csv(data_dir)
    # print(f"[info@main] -> All predictions completed and saved to {data_dir}.")