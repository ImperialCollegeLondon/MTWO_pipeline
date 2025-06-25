from collections import Counter
import numpy as np
import os
import sys
from loguru import logger

# Add the parent directory to the Python path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cache_dir
from dataAugmenter.balance_classes import balance_classes
import random
from typing import Dict, Union, Tuple, Optional

# Configure loguru logger
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

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
    logger.info(f"Class distribution after balancing: {Counter(balanced_y)}")

    np.save(os.path.join(cache_dir, 'balanced_data.npy'), balanced_X)
    np.save(os.path.join(cache_dir, 'balanced_labels.npy'), balanced_y)
    logger.info(f"Balanced data saved as cache.")

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
    logger.info(f"Class distribution after balancing: {Counter(balanced_y)}")

    np.save(os.path.join(cache_dir, 'balanced_data.npy'), balanced_X)
    np.save(os.path.join(cache_dir, 'balanced_labels.npy'), balanced_y)
    logger.info(f"Balanced data saved as cache.")

    return balanced_X, balanced_y


import matplotlib.pyplot as plt

def visualize_augmentation_comparison(
    original_acc: np.ndarray,
    original_labels: np.ndarray,
    augmented_acc: np.ndarray,
    augmented_labels: np.ndarray,
    mode: str = "MO",
    comparison_type: str = "segment",
    segment_length: int = 200,
    start_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (18, 12)
):
    """
    Visualize accelerometer data before and after augmentation.
    
    Args:
        original_acc: Original accelerometer data (numpy array with shape (n_samples, time_steps, 3))
        original_labels: Original labels corresponding to original_acc
        augmented_acc: Augmented accelerometer data (numpy array with shape (n_samples, time_steps, 3))
        augmented_labels: Augmented labels corresponding to augmented_acc
        mode: "MO" or "MTWO" for different classification modes
        comparison_type: "full", "segment", or "random" for data selection
        segment_length: Number of samples to visualize (for segment/random modes)
        start_idx: Starting index for segment mode (if None, defaults to 0)
        figsize: Figure size for the plot
    """
    
    # Debug: Print data shapes and time steps
    logger.info(f"Data shapes - Original: {original_acc.shape}, Augmented: {augmented_acc.shape}")
    if len(original_acc) > 0:
        logger.debug(f"Original data time steps: {original_acc[0].shape[0]}")
    if len(augmented_acc) > 0:
        logger.debug(f"Augmented data time steps: {augmented_acc[0].shape[0]}")
    
    # Select data based on comparison type
    if comparison_type == "full":
        orig_X_vis, orig_y_vis = original_acc, original_labels
        aug_X_vis, aug_y_vis = augmented_acc, augmented_labels
        title_suffix = "Full Dataset"
    elif comparison_type == "segment":
        if start_idx is None:
            start_idx = 0
        end_idx = min(start_idx + segment_length, len(original_acc))
        orig_X_vis = original_acc[start_idx:end_idx]
        orig_y_vis = original_labels[start_idx:end_idx]
        
        # For augmented data, take the same segment length from the beginning
        aug_end_idx = min(segment_length, len(augmented_acc))
        aug_X_vis = augmented_acc[:aug_end_idx]
        aug_y_vis = augmented_labels[:aug_end_idx]
        title_suffix = f"Segment [{start_idx}:{end_idx}]"
    elif comparison_type == "random":
        # Random selection from original data
        orig_indices = random.sample(range(len(original_acc)), min(segment_length, len(original_acc)))
        orig_X_vis = original_acc[orig_indices]
        orig_y_vis = original_labels[orig_indices]
        
        # Random selection from augmented data
        aug_indices = random.sample(range(len(augmented_acc)), min(segment_length, len(augmented_acc)))
        aug_X_vis = augmented_acc[aug_indices]
        aug_y_vis = augmented_labels[aug_indices]
        title_suffix = f"Random Sample ({segment_length} points)"
    else:
        raise ValueError("comparison_type must be 'full', 'segment', or 'random'")
    
    # Create subplots - 3 rows, 3 columns for better layout
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(f'Movement vs Other Data Augmentation Comparison - {mode} Mode - {title_suffix}', fontsize=16)
    
    # Colors for different classes
    colors = {'M': 'green', 'O': 'orange'}
    if mode == "MTWO":
        colors = {'M': 'green', 'T': 'blue', 'W': 'black', 'O': 'orange'}
    
    # Plot original data distribution
    orig_counter = Counter(orig_y_vis)
    axes[0, 0].bar(orig_counter.keys(), orig_counter.values(), 
                    color=[colors.get(k, 'gray') for k in orig_counter.keys()])
    axes[0, 0].set_title('Original Class Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # Plot augmented data distribution
    aug_counter = Counter(aug_y_vis)
    axes[0, 1].bar(aug_counter.keys(), aug_counter.values(), 
                    color=[colors.get(k, 'gray') for k in aug_counter.keys()])
    axes[0, 1].set_title('Augmented Class Distribution')
    axes[0, 1].set_ylabel('Count')
    
    # Plot comparison bar chart
    all_classes = sorted(set(list(orig_counter.keys()) + list(aug_counter.keys())))
    x_pos = np.arange(len(all_classes))
    orig_counts = [orig_counter.get(cls, 0) for cls in all_classes]
    aug_counts = [aug_counter.get(cls, 0) for cls in all_classes]
    
    width = 0.35
    axes[0, 2].bar(x_pos - width/2, orig_counts, width, label='Original', alpha=0.7, color='lightblue')
    axes[0, 2].bar(x_pos + width/2, aug_counts, width, label='Augmented', alpha=0.7, color='lightcoral')
    axes[0, 2].set_title('Class Distribution Comparison')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(all_classes)
    axes[0, 2].legend()
    
    # Plot original accelerometer data samples (X, Y, Z axes)
    axes_names = ['X-axis', 'Y-axis', 'Z-axis']
    for axis_idx in range(3):
        # Original data
        sample_count = 0
        for class_label, color in colors.items():
            if class_label in orig_y_vis:
                class_data = orig_X_vis[orig_y_vis == class_label]
                if len(class_data) > 0:
                    # Plot first few samples of each class
                    for j in range(min(3, len(class_data))):
                        sample = class_data[j]
                        # Debug: Print sample shape
                        if sample_count == 0 and axis_idx == 0:
                            logger.info(f"Original sample {j} shape for class {class_label}: {sample.shape}")
                        
                        # Plot the entire time series
                        time_steps = np.arange(len(sample))
                        axes[1, axis_idx].plot(time_steps, sample[:, axis_idx], color=color, alpha=0.6, 
                                               linewidth=1, label=f'{class_label}' if j == 0 else "")
                        sample_count += 1
        
        axes[1, axis_idx].set_title(f'Original Data ({axes_names[axis_idx]})')
        axes[1, axis_idx].set_xlabel('Time Steps')
        axes[1, axis_idx].set_ylabel('Acceleration')
        axes[1, axis_idx].legend()
        axes[1, axis_idx].grid(True, alpha=0.3)
        # Set x-axis limits to show full time series
        if len(orig_X_vis) > 0:
            max_time_steps = max([sample.shape[0] for sample in orig_X_vis])
            axes[1, axis_idx].set_xlim(0, max_time_steps)
        
        # Augmented data
        sample_count = 0
        for class_label, color in colors.items():
            if class_label in aug_y_vis:
                class_data = aug_X_vis[aug_y_vis == class_label]
                if len(class_data) > 0:
                    # Plot first few samples of each class
                    for j in range(min(3, len(class_data))):
                        sample = class_data[j]
                        # Debug: Print sample shape
                        if sample_count == 0 and axis_idx == 0:
                            logger.info(f"Augmented sample {j} shape for class {class_label}: {sample.shape}")
                        
                        # Plot the entire time series
                        time_steps = np.arange(len(sample))
                        axes[2, axis_idx].plot(time_steps, sample[:, axis_idx], color=color, alpha=0.6, 
                                               linewidth=1, label=f'{class_label}' if j == 0 else "")
                        sample_count += 1
        
        axes[2, axis_idx].set_title(f'Augmented Data ({axes_names[axis_idx]})')
        axes[2, axis_idx].set_xlabel('Time Steps')
        axes[2, axis_idx].set_ylabel('Acceleration')
        axes[2, axis_idx].legend()
        axes[2, axis_idx].grid(True, alpha=0.3)
        # Set x-axis limits to show full time series
        if len(aug_X_vis) > 0:
            max_time_steps = max([sample.shape[0] for sample in aug_X_vis])
            axes[2, axis_idx].set_xlim(0, max_time_steps)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'movement_other_augmentation_comparison_{mode}_{comparison_type}.png'
    plot_path = os.path.join(cache_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {plot_path}")

    # Print detailed statistics
    logger.info("=== Movement vs Other Augmentation Statistics ===")
    logger.info(f"Original data shape: {original_acc.shape}")
    logger.info(f"Augmented data shape: {augmented_acc.shape}")
    logger.info(f"Original class distribution: {Counter(original_labels)}")
    logger.info(f"Augmented class distribution: {Counter(augmented_labels)}")
    
    # Calculate and log augmentation ratios
    for class_label in ['M', 'O'] if mode == "MO" else ['M', 'T', 'W', 'O']:
        orig_count = orig_counter.get(class_label, 0)
        aug_count = aug_counter.get(class_label, 0)
        if orig_count > 0:
            ratio = aug_count / orig_count
            logger.info(f"Class {class_label}: Original={orig_count}, Augmented={aug_count}, Ratio={ratio:.2f}")
    
    plt.show()
    