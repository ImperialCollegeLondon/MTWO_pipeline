import numpy as np
import pandas as pd
import gc
from joblib import Parallel, delayed
from tqdm import tqdm

from featureExtractor.features import compute_features
from featureExtractor.features import compute_features_1
from featureExtractor.features import compute_features_MO

def extract_features_batch(batch, batch_labels, mode='MTWO'):
    """
    Extract features from a batch of 3D accelerometer data with error handling.
    For 'T' type data use compute_features_1, for others use compute_features.
    
    Args:
        batch: Batch of 3D accelerometer data with shape (batch_size, window_size, n_features)
        batch_labels: Corresponding labels for the batch
        
    Returns:
        NumPy array of features with shape (batch_size, n_extracted_features)
    """
    batch_features = []
    for i, (window, label) in enumerate(zip(batch, batch_labels)):
        try:
            df = pd.DataFrame(window, columns=['X', 'Y', 'Z'])
                # if label == 'T':
                #     features = compute_features_1(df)
                # else:
                #     features = compute_features(df)
            if mode == 'MO':
                features = compute_features_MO(df)
            else:  # MTWO mode
                features = compute_features(df)
            batch_features.append(features)
        except Exception as e:
            raise RuntimeError(f"Error processing batch {i}: {e}")
    return np.array(batch_features)

def extract_features(balanced_X, balanced_y, mode='MTWO', batch_size=100, n_jobs=-1, show_details=False):
    print("[info@extract_features] -> Extracting features from data...")
    n_samples = balanced_X.shape[0]
    batches = [balanced_X[i:min(i+batch_size, n_samples)] for i in range(0, n_samples, batch_size)]
    batch_labels = [balanced_y[i:min(i+batch_size, n_samples)] for i in range(0, n_samples, batch_size)]

    # Extract features in parallel
    features = Parallel(n_jobs=n_jobs)(
        delayed(extract_features_batch)(batch, labels, mode) for batch, labels in tqdm(zip(batches, batch_labels), total=len(batches), desc="[info@extract_features] -> Processing batches", leave=False, unit="batch")
    )

    # Combine all batches
    X_features_ori = np.vstack(features)

    # 当窗口大小小于4时，需要处理NaN值
    X_features = np.nan_to_num(X_features_ori, nan=0.0, posinf=0.0, neginf=0.0)
    # if X_features != X_features_ori:
    #     print("NaN values detected! Replaced with 0.0")

    # Save the extracted features and labels
    # features_path = os.path.join(cache_dir, "features.npy")
    # labels_path = os.path.join(cache_dir, "labels.npy")
    # np.save(features_path, X_features)
    # np.save(labels_path, balanced_y)
    # print(f"Features saved to {features_path}")
    # print(f"Labels saved to {labels_path}")
    if show_details:
        print("\n---------------- Features Details ----------------")
        print("@extract_features")
        print("Features shape:", X_features.shape)
        # Show sample of the extracted features
        print("\nSample of extracted features (first 3 samples):")
        print(X_features[:3])

        # Show feature statistics
        print("\nFeature statistics:")
        print(f"Min: {X_features.min()}, Max: {X_features.max()}")
        print(f"Mean: {X_features.mean():.4f}, Std: {X_features.std():.4f}")
        print('--------------------------------------------------\n')
    # Memory cleanup
    del batches, features
    gc.collect()

    return np.array(X_features)