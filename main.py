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

def main(mode):
    """Main function to run the entire ML pipeline."""
    random.seed(SEED)  # Set random seed for reproducibility
    print()

    if mode == 'MTWO':
        print('Training started. Mode: MTWO.')
        # 1. Load Data
        # - Either load data from custom dataset:
        # movement = load_data(movement_dir, useFilter=True)
        # transport = load_data(transport_dir, useFilter=True)
        # walking = load_data(walking_dir, useFilter=True)
        # other = load_data(others_dir, useFilter=True)
        # - Or uncomment to Load data from original sources (AX and LAB data):
        movement, transport, walking, other = load_data_from_original_sources(loadNewTransport=False)
        print(f"[info@main] -> All data Loaded: Movement: {len(movement)}, Transport: {len(transport)}, Walking: {len(walking)}, Other: {len(other)}")
        print('----------------------------------------------------\n')

        # 2. Data Augmentation
        data, labels = augment_data(movement, transport, walking, other)
        print(f"[info@main] -> Data Augmentation Completed: {len(data)} samples with {len(set(labels))} unique labels.")
        print('----------------------------------------------------\n')
    
    elif mode == 'MO':
        print('Training started. Mode: MO.')
        # 1. Load Data
        movement_list = load_data(movement_dir, useFilter=True) # Load data from custom dataset, return: list(pd.Dataframe,)
        # other = load_data(others_dir, useFilter=True) # Load data from custom dataset
        _, _, _, other = load_data_from_original_sources(loadNewTransport=False) # Load data from original sources (AX and LAB data)

        # Convert custom movement data to the same format as original sources data
        # Apply sliding window to convert DataFrames to windowed arrays
        from dataTransformer.sliding_window import splitIntoOverlappingWindows
        movement_windowed = []
        for df in movement_list:
            windows = splitIntoOverlappingWindows(df)
            movement_windowed.extend(windows)
        
        # Convert to numpy arrays maintaining the 3D structure (samples, time_steps, features)
        movement = np.array([window.values for window in movement_windowed])

        print(f"[info@main] -> All data Loaded: Movement: {len(movement)}, Other: {len(other)}")
        print(f"[info@main] -> Movement shape: {movement.shape}, Other shape: {other.shape}")
        print('----------------------------------------------------\n')


        # 2. Data Augmentation
        data, labels = augment_data_MO(movement, other)
        print(f"[info@main] -> Data Augmentation Completed: {len(data)} samples with {len(set(labels))} unique labels.")
        print('----------------------------------------------------\n')


    # 3. Feature Extraction
    X_features = extract_features(data, labels, mode=mode)
    print(f"[info@main] -> Feature Extraction Completed: Extracted {X_features.shape[1]} features from {len(data)} samples.")
    print('----------------------------------------------------\n')


    # 4. Data processing: encoding and scaling
    y_labels = encoder.encode(labels) # encode the labels from str to int
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels)
    X_train_scaled, X_test_scaled = scaler.scale(X_train, X_test)  # scale the features
    print(f"[info@main] -> Data Processing Completed: Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    print('----------------------------------------------------\n')


    # 5. Dimensionality Reduction
    X_train_pca, X_test_pca, pca_model = PCA.pca(X_train_scaled, X_test_scaled, n_components=0.95, vis=False)
    print(f"[info@main] -> Dimensionality Reduction Completed: Reduced features from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]} dimensions.")
    print('----------------------------------------------------\n')
    

    # 6. Train and Evaluate
    results_dict, (best_model_name, best_accuracy, best_aurc) = train_and_evaluate_all(X_train_pca, X_test_pca, y_train, y_test, getBest=True, mode=mode)
    save_comparison(results_dict, (best_model_name, best_accuracy, best_aurc))
    print(f"[info@main] -> Training and Evaluation Completed. Best Model: {best_model_name} with accuracy {best_accuracy:.2f}.")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Print the time of the last run
    print(f"\nScript last run at {current_time}")

if __name__ == '__main__':
    main(mode='MO')