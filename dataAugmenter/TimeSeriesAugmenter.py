import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import random

from config import SEED

class TimeSeriesAugmenter:
    """Time series data augmentation for accelerometer data."""
    
    def __init__(self, random_state=SEED):
        """
        Initialize the augmenter with optional random state.
        
        Args:
            random_state: Integer seed for reproducibility
        """
    
    def add_gaussian_noise(self, X, noise_level=(0.001, 0.05)):
        """
        Add gaussian noise to the time series data.
        
        Args:
            X: Time series data with shape (n_samples, window_size, n_features)
            noise_level: Tuple of (min_std, max_std) for the noise
        
        Returns:
            Augmented data with same shape as X
        """
        noise_std = np.random.uniform(noise_level[0], noise_level[1])
        noise = np.random.normal(0, noise_std, size=X.shape)
        return X + noise
    
    def scale(self, X, scale_range=(0.8, 1.2)):
        """
        Multiply time series by a random scalar.
        
        Args:
            X: Time series data with shape (n_samples, window_size, n_features)
            scale_range: Range of scaling factors
        
        Returns:
            Scaled data with same shape as X
        """
        scales = np.random.uniform(scale_range[0], scale_range[1], size=(X.shape[0], 1, X.shape[2]))
        return X * scales
    
    def magnitude_warp(self, X, sigma=0.2, knot=4):
        """
        Apply random magnitude warping.
        
        Args:
            X: Time series data with shape (n_samples, window_size, n_features)
            sigma: Standard deviation of the random knots
            knot: Number of knots for the spline
        
        Returns:
            Warped data with same shape as X
        """
        augmented = np.zeros(X.shape)
        for i in range(X.shape[0]):
            window_size = X.shape[1]
            knot_points = np.linspace(0, window_size-1, knot+2)[1:-1]
            knot_values = np.random.normal(1.0, sigma, knot)
            
            # Create spline with boundary knots
            x_points = np.arange(0, window_size)
            all_knot_points = np.concatenate(([0], knot_points, [window_size-1]))
            all_knot_values = np.concatenate(([1.0], knot_values, [1.0]))
            
            warping = CubicSpline(all_knot_points, all_knot_values)(x_points)
            
            # Apply warping separately to each channel
            for c in range(X.shape[2]):
                augmented[i, :, c] = X[i, :, c] * warping
                
        return augmented
    
    def time_shift(self, X, shift_range=(-10, 10)):
        """
        Shift the time series randomly along time axis.
        
        Args:
            X: Time series data with shape (n_samples, window_size, n_features)
            shift_range: Range of shifts (in time steps)
        
        Returns:
            Time shifted data with same shape as X
        """
        augmented = np.zeros(X.shape)
        for i in range(X.shape[0]):
            shift = np.random.randint(shift_range[0], shift_range[1])
            for c in range(X.shape[2]):
                augmented[i, :, c] = np.roll(X[i, :, c], shift)
                
        return augmented
    
    def rotate(self, X, rotation_range=(-0.1, 0.1)):
        """
        Apply small 3D rotations to the XYZ accelerometer data.
        
        Args:
            X: Time series data with shape (n_samples, window_size, 3)
            rotation_range: Range of rotation angles in radians
        
        Returns:
            Rotated data with same shape as X
        """
        if X.shape[2] != 3:
            raise ValueError("Rotation only works with 3D features (X, Y, Z)")
            
        augmented = np.zeros(X.shape)
        for i in range(X.shape[0]):
            # Random rotation angles
            theta_x = np.random.uniform(rotation_range[0], rotation_range[1])
            theta_y = np.random.uniform(rotation_range[0], rotation_range[1])
            theta_z = np.random.uniform(rotation_range[0], rotation_range[1])
            
            # Rotation matrices
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(theta_x), -np.sin(theta_x)],
                          [0, np.sin(theta_x), np.cos(theta_x)]])
            
            Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                          [0, 1, 0],
                          [-np.sin(theta_y), 0, np.cos(theta_y)]])
            
            Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                          [np.sin(theta_z), np.cos(theta_z), 0],
                          [0, 0, 1]])
            
            R = np.dot(Rz, np.dot(Ry, Rx))  # Combine rotations
            
            # Apply rotation to each time step
            for t in range(X.shape[1]):
                augmented[i, t, :] = np.dot(R, X[i, t, :])
                
        return augmented
    
    def augment(self, X, num_augmented=1, methods=None):
        """
        Generate augmented versions of the input data using multiple methods.
        
        Args:
            X: Time series data with shape (n_samples, window_size, n_features)
            num_augmented: Number of augmented copies to generate per sample
            methods: List of augmentation methods to use, or None for all
                    Options: 'noise', 'scale', 'magnitude_warp', 'time_shift', 'rotate'
        
        Returns:
            Augmented data with shape (n_samples * (num_augmented + 1), window_size, n_features)
        """
        if methods is None:
            if X.shape[2] == 3:  # If data has 3 features (X,Y,Z)
                methods = ['noise', 'scale', 'magnitude_warp', 'time_shift', 'rotate']
            else:
                methods = ['noise', 'scale', 'magnitude_warp', 'time_shift']
        
        aug_data = [X]  # Start with the original data
        
        for _ in range(num_augmented):
            X_aug = X.copy()
            
            # Apply a random combination of augmentations
            aug_methods = random.sample(methods, k=random.randint(1, len(methods)))
            
            for method in aug_methods:
                if method == 'noise':
                    X_aug = self.add_gaussian_noise(X_aug)
                elif method == 'scale':
                    X_aug = self.scale(X_aug)
                elif method == 'magnitude_warp':
                    X_aug = self.magnitude_warp(X_aug)
                elif method == 'time_shift':
                    X_aug = self.time_shift(X_aug)
                elif method == 'rotate' and X.shape[2] == 3:
                    X_aug = self.rotate(X_aug)
            
            aug_data.append(X_aug)
        
        return np.vstack(aug_data)