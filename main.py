'''
@ Author: Yufeng NA
@ Imperial College London
@ Date: June 5, 2025
@ Description: Main script to run the entire machine learning pipeline for activity recognition.
# ------------------Main Script-----------------------------
# This script orchestrates the entire machine learning pipeline, including data loading, augmentation,
# feature extraction, data processing, dimensionality reduction, and model training and evaluation.
'''

from sklearn.model_selection import train_test_split
import random
import datetime
import numpy as np
import sys

import pandas as pd
from loguru import logger

from config import *

from dataLoader.load_data import load_data
from dataLoader.load_data import load_data_from_original_sources
from dataLoader.load_data import df2array
from dataAugmenter.augment_data import augment_data
from dataAugmenter.augment_data import augment_data_MO
from featureExtractor.extract_features import extract_features

from dataTransformer import encoder
from dataTransformer import scaler
from dataTransformer import PCA

from trainAndEvaluation.train_and_evaluate_all import train_and_evaluate_all
from trainAndEvaluation.train_and_evaluate_all import save_comparison
sys.path.insert(0, '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/Data/my_data/compare')
import transform  # Import the transform module to use its functions

# Configure loguru logger with colors
logger.remove()  # Remove default handler
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    "logs/main_{time:YYYY-MM-DD}.log", 
    rotation="1 day", 
    retention="7 days", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    colorize=False  # 文件输出不需要颜色
)

# 自定义级别颜色
logger.level("INFO", color="<blue>")
logger.level("SUCCESS", color="<green>")
logger.level("WARNING", color="<yellow>")
logger.level("ERROR", color="<red>")
logger.level("DEBUG", color="<cyan>")

def array_to_dataframe_list(data_array):
    """
    Convert 3D numpy array to list of pandas DataFrames
    
    Args:
        data_array: 3D numpy array with shape (samples, time_steps, features)
        
    Returns:
        list of pandas DataFrames
    """
    dataframe_list = []
    for sample in data_array:
        df = pd.DataFrame(sample, columns=['accelerationX', 'accelerationY', 'accelerationZ'])
        dataframe_list.append(df)
    return dataframe_list

def main(mode):
    """Main function to run the entire ML pipeline."""
    random.seed(SEED)  # Set random seed for reproducibility
    logger.info(f'Training started. Mode: {mode}.')
    logger.info("----------------------------------------------------")

    if mode == 'MTWO':
        logger.info("Step #1 Loading data...")
        # 1. Load Data
        # - Either load data from custom dataset:
        # movement = load_data(movement_dir, useFilter=True)
        # transport = load_data(transport_dir, useFilter=True)
        # walking = load_data(walking_dir, useFilter=True)
        # other = load_data(others_dir, useFilter=True)
        # - Or uncomment to Load data from original sources (AX and LAB data):
        movement, transport, walking, other = load_data_from_original_sources(loadNewTransport=False)
        logger.info(f"All data Loaded: Movement: {len(movement)}, Transport: {len(transport)}, Walking: {len(walking)}, Other: {len(other)}")
        logger.info('----------------------------------------------------\n')

        # 2. Data Augmentation
        data, labels = augment_data(movement, transport, walking, other)
        logger.info(f"Data Augmentation Completed: {len(data)} samples with {len(set(labels))} unique labels.")
        logger.info('----------------------------------------------------\n')
    
    elif mode == 'MO':
        logger.info("Step #1 Loading data...")
        # 1. Load Data
        movement_list = load_data(movement_dir, useFilter=True) # Load data from custom dataset, return: list(pd.Dataframe,)
        other_list = load_data(others_dir, useFilter=True) # Load data from custom dataset
        # _, _, _, other = load_data_from_original_sources(loadNewTransport=False) # Load data from original sources (AX and LAB data)


        # Mapping other from AX to Apple Watch # Convert numpy array to DataFrame list for mapping
        # other_dataframes = array_to_dataframe_list(other)
        # other_mapped = transform.map_from_custom_data(other_dataframes)
        # logger.info("Other data mapped from AX format to Apple Watch.")
        # other = np.array([df[['accelerationX', 'accelerationY', 'accelerationZ']].values for df in other_mapped]) # Convert mapped DataFrames back to numpy array format


        # Convert custom data to the same format as original sources data
        # Apply sliding window to convert DataFrames to windowed arrays
        from dataTransformer.sliding_window import splitIntoOverlappingWindows
        movement_windowed, other_windowed = [], []
        for df in movement_list:
            windows = splitIntoOverlappingWindows(df)
            movement_windowed.extend(windows)
        for df in other_list:
            windows = splitIntoOverlappingWindows(df)
            other_windowed.extend(windows)
        
        # Convert to numpy arrays maintaining the 3D structure (samples, time_steps, features)
        movement = np.array([window.values for window in movement_windowed])
        other = np.array([window.values for window in other_windowed])

        logger.info(f"All data Loaded: Movement: {len(movement)}, Other: {len(other)}")
        logger.info(f"Movement shape: {movement.shape}, Other shape: {other.shape}")
        logger.info('----------------------------------------------------\n')


        # 2. Data Augmentation
        logger.info("Step #2 Data Augmenting...")
        data, labels = augment_data_MO(movement, other)
        logger.info(f"Data Augmentation Completed: {len(data)} samples with {len(set(labels))} unique labels.")
        logger.info('----------------------------------------------------\n')


    # 3. Feature Extraction
    logger.info("Step #3 Feature Extraction...")
    X_features = extract_features(data, labels, mode=mode)
    logger.info(f"Feature Extraction Completed: Extracted {X_features.shape[1]} features from {len(data)} samples.")
    logger.info('----------------------------------------------------\n')


    # 4. Data processing: encoding and scaling
    logger.info("Step #4 Data Processing...")
    y_labels = encoder.encode(labels) # encode the labels from str to int
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2)
    X_train_scaled, X_test_scaled = scaler.scale(X_train, X_test)  # scale the features
    logger.info(f"Data Processing Completed: Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    logger.info('----------------------------------------------------\n')


    # 5. Dimensionality Reduction
    logger.info("Step #5 Dimensionality Reduction...")
    X_train_pca, X_test_pca = PCA.pca(X_train_scaled, X_test_scaled, n_components=0.95, vis=False)
    logger.info(f"Dimensionality Reduction Completed: Reduced features from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]} dimensions.")
    logger.info('----------------------------------------------------\n')
    
    # 6. Train and Evaluate
    if mode == 'MO':
        class_names = ['Others', 'Movement']
    else:  # MTWO
        class_names = ['Movement', 'Transport', 'Walking', 'Others']
    
    logger.info("Step #6 Training and Evaluation...")
    results_dict, (best_model_name, best_accuracy, best_aurc) = train_and_evaluate_all(
        X_train_pca, X_test_pca, y_train, y_test, 
        getBest=True, mode=mode, class_names=class_names, save_confusion_matrices=True
    )
    save_comparison(results_dict, (best_model_name, best_accuracy, best_aurc))
    logger.info(f"Training and Evaluation Completed. Best Model: {best_model_name} with accuracy {best_accuracy:.2f}.")
    
    # 7. Generate confusion matrix comparison
    logger.info("\nStep #7 Generating confusion matrix comparison...")
    from trainAndEvaluation.confusion_matrix_utils import compare_models_confusion_matrices
    models_to_compare = ['xgboost', 'rf', 'mlp'] if mode == 'MTWO' else ['xgboost2', 'rf', 'mlp']
    compare_models_confusion_matrices(X_test_pca, y_test, models_to_compare, 
                                    class_names=class_names, mode=mode)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Print the time of the last run
    logger.info(f"\nScript last run at {current_time}")

if __name__ == '__main__':
    main(mode='MO')

    # Plot feature importance for XGBoost model
    # import joblib
    # import xgboost as xgb
    # import matplotlib.pyplot as plt
    # booster = joblib.load('/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project/saved_models/ML/xgboost.pkl')
    # # 画出基于 "gain" 的特征重要性条形图
    # xgb.plot_importance(booster, 
    #                     importance_type='gain', 
    #                     max_num_features=15,  # 显示前15个
    #                     height=0.5,
    #                     xlabel='Average Gain',
    #                     title='Feature Importance (by Gain)',
    #                     grid=False)

    # plt.tight_layout()
    # plt.show()

    logger.success('Pipeline completed successfully!')