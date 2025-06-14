from collections import Counter
import numpy as np
import os

from config import cache_dir
from dataAugmenter.balance_classes import balance_classes

def augment_data(movement, transport, walking, other):
    # Balance classes
    X_dict = {
        'M': movement,
        'O': other,
        'T': transport,
        # 'T1': None,
        'W': walking
    }
    y_dict = {
        'M': np.array(['M'] * len(movement)),
        'O': np.array(['O'] * len(other)),
        'T': np.array(['T'] * len(transport)),
        # 'T1': np.array(['T1'] * len(None)),
        'W': np.array(['W'] * len(walking))
    }


    balanced_X, balanced_y = balance_classes(X_dict, y_dict)
    print(f"[info@augment_data] -> {Counter(balanced_y)}")

    np.save(os.path.join(cache_dir, 'balanced_data.npy'), balanced_X)
    np.save(os.path.join(cache_dir, 'balanced_labels.npy'), balanced_y)
    print(f"[info@augment_data] -> Balanced data saved as cache.")

    return balanced_X, balanced_y

def augment_data_MO(movement, other):
    '''
    Input papareters type:
    movment: np.array
    other: np.array
    '''
    # Balance classes
    X_dict = {
        'M': movement,
        'O': other,
    }
    y_dict = {
        'M': np.array(['M'] * len(movement)),
        'O': np.array(['O'] * len(other)),
    }

    balanced_X, balanced_y = balance_classes(X_dict, y_dict)
    print(f"[info@augment_data] -> {Counter(balanced_y)}")

    np.save(os.path.join(cache_dir, 'balanced_data.npy'), balanced_X)
    np.save(os.path.join(cache_dir, 'balanced_labels.npy'), balanced_y)
    print(f"[info@augment_data] -> Balanced data saved as cache.")

    return balanced_X, balanced_y