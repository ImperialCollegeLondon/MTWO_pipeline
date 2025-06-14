import numpy as np

from dataAugmenter.TimeSeriesAugmenter import TimeSeriesAugmenter 
from config import SEED

def balance_classes(X_dict, y_dict, target_count=None):
    """
    Balance classes by augmenting minority classes to match the majority class.
    
    Args:
        X_dict: Dictionary mapping class labels to data arrays (key:str, value:np.array)
        y_dict: Dictionary mapping class labels to label arrays (key:str, value:int)
        target_count: Target count for each class, or None to use the maximum class count
    
    Returns:
        Tuple of (balanced_X, balanced_y)
    """
    # Find the target count (max class size if not specified)
    if target_count is None:
        target_count = max(X.shape[0] for X in X_dict.values())
    
    augmenter = TimeSeriesAugmenter(random_state=SEED)
    balanced_data = []
    balanced_labels = []
    
    for label, X in X_dict.items():
        if X.shape[0] < target_count:
            # Calculate how many augmented samples needed
            num_aug = int(np.ceil((target_count - X.shape[0]) / X.shape[0]))
            X_balanced = augmenter.augment(X, num_augmented=num_aug)
            # Take only what we need to reach the target
            X_balanced = X_balanced[:target_count]
            y_balanced = np.full(target_count, y_dict[label][0])
        else:
            # If we already have enough samples, just take what we need
            X_balanced = X[:target_count]
            y_balanced = y_dict[label][:target_count]
            
        balanced_data.append(X_balanced)
        balanced_labels.append(y_balanced)

        # 在返回之前，使用更安全的方式合并
    try:
        combined_data = np.vstack(balanced_data)
        combined_labels = np.concatenate(balanced_labels)
        return combined_data, combined_labels
    except ValueError as e:
        print(f"Error combining data: {e}")
        print("Data shapes:")
        for i, data in enumerate(balanced_data):
            print(f"  balanced_data[{i}]: {data.shape}\n")
        raise