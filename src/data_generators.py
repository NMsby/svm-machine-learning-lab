"""
Data Generation Module for SVM Lab

This module provides functions to generate various types of datasets
for testing and comparing SVM implementations.

Author: Nelson Masbayi
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def generate_linearly_separable_data(
        n_samples: int = 100,
        random_state: Optional[int] = 42,
        class_sep: float = 2.0,
        noise_std: float = 0.5,
        center_distance: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D linearly separable binary classification dataset.

    Creates two Gaussian clusters that are linearly separable with
    adjustable separation distance and noise levels.

    Parameters:
    -----------
    n_samples : int, default=100
        Total number of samples to generate
    random_state : int or None, default=42
        Random seed for reproducibility. If None, no seed is set.
    class_sep : float, default=2.0
        Separation factor between class centers
    noise_std : float, default=0.5
        Standard deviation of Gaussian noise for each cluster
    center_distance : float, default=2.0
        Base distance between cluster centers

    Returns:
    --------
    X : np.ndarray of shape (n_samples, 2)
        Feature matrix with 2D coordinates
    y : np.ndarray of shape (n_samples,)
        Binary labels (-1 for class 1, +1 for class 2)
    """

    # Input validation
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if n_samples % 2 != 0:
        warnings.warn("n_samples is odd, will create n_samples+1 total samples")
        n_samples += 1

    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate samples per class
    samples_per_class = n_samples // 2

    # Define cluster centers with specified separation
    center1 = np.array([center_distance * class_sep, center_distance * class_sep])
    center2 = np.array([-center_distance * class_sep, -center_distance * class_sep])

    # Generate Class +1 (centered around positive quadrant)
    X1 = np.random.normal(center1, noise_std, (samples_per_class, 2))
    y1 = np.ones(samples_per_class)

    # Generate Class -1 (centered around negative quadrant)
    X2 = np.random.normal(center2, noise_std, (samples_per_class, 2))
    y2 = -np.ones(samples_per_class)

    # Combine datasets
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])

    # Shuffle the data to avoid any ordering bias
    shuffle_indices = np.random.permutation(len(X))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    return X, y


def generate_nonlinear_data(
        n_samples: int = 200,
        random_state: Optional[int] = 42,
        inner_radius: float = 1.0,
        outer_radius_min: float = 1.5,
        outer_radius_max: float = 2.5,
        noise_factor: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D non-linearly separable binary classification dataset.

    Creates a circular/ring pattern where one class forms an inner circle
    and the other class forms an outer ring, making linear separation impossible.

    Parameters:
    -----------
    n_samples : int, default=200
        Total number of samples to generate (split equally between classes)
    random_state : int or None, default=42
        Random seed for reproducibility. If None, no seed is set.
    inner_radius : float, default=1.0
        Maximum radius for inner circle (class -1)
    outer_radius_min : float, default=1.5
        Minimum radius for outer ring (class +1)
    outer_radius_max : float, default=2.5
        Maximum radius for outer ring (class +1)
    noise_factor : float, default=0.1
        Amount of noise to add to the radial distances

    Returns:
    --------
    X : np.ndarray of shape (n_samples, 2)
        Feature matrix with 2D coordinates
    y : np.ndarray of shape (n_samples,)
        Binary labels (-1 for inner circle, +1 for outer ring)
    """

    # Input validation
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if outer_radius_min <= inner_radius:
        raise ValueError("outer_radius_min must be greater than inner_radius")
    if outer_radius_max <= outer_radius_min:
        raise ValueError("outer_radius_max must be greater than outer_radius_min")

    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate samples per class
    samples_per_class = n_samples // 2

    # Generate inner circle (class -1)
    # Random radius between 0 and inner_radius
    r1 = np.random.uniform(0, inner_radius, samples_per_class)
    # Add noise to radius
    r1 += np.random.normal(0, noise_factor * inner_radius, samples_per_class)
    r1 = np.clip(r1, 0, None)  # Ensure non-negative radius

    # Random angles
    theta1 = np.random.uniform(0, 2 * np.pi, samples_per_class)

    # Convert to Cartesian coordinates
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = -np.ones(samples_per_class)

    # Generate outer ring (class +1)
    # Random radius between outer_radius_min and outer_radius_max
    r2 = np.random.uniform(outer_radius_min, outer_radius_max, samples_per_class)
    # Add noise to radius
    r2 += np.random.normal(0, noise_factor * outer_radius_max, samples_per_class)
    r2 = np.clip(r2, outer_radius_min, None)  # Ensure minimum radius

    # Random angles
    theta2 = np.random.uniform(0, 2 * np.pi, samples_per_class)

    # Convert to Cartesian coordinates
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    y2 = np.ones(samples_per_class)

    # Combine datasets
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])

    # Shuffle the data to avoid ordering bias
    shuffle_indices = np.random.permutation(len(X))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    return X, y


def get_dataset_info(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Get comprehensive information about a dataset.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Label vector

    Returns:
    --------
    info : dict
        Dictionary containing dataset statistics
    """

    unique_labels, counts = np.unique(y, return_counts=True)

    info = {
        'n_samples': len(X),
        'n_features': X.shape[1] if X.ndim > 1 else 1,
        'n_classes': len(unique_labels),
        'class_labels': unique_labels.tolist(),
        'class_counts': counts.tolist(),
        'class_balance': dict(zip(unique_labels, counts)),
        'feature_ranges': {
            'min': X.min(axis=0).tolist(),
            'max': X.max(axis=0).tolist(),
            'mean': X.mean(axis=0).tolist(),
            'std': X.std(axis=0).tolist()
        }
    }

    return info


def validate_dataset(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Validate that a dataset is properly formatted for binary classification.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Label vector

    Returns:
    --------
    bool
        True if dataset is valid, raises exception otherwise

    Raises:
    -------
    ValueError
        If the dataset format is invalid
    """

    # Check basic shapes
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got {X.ndim}D")

    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got {y.ndim}D")

    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")

    # Check for binary classification
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError(f"Binary classification requires exactly 2 classes, got {len(unique_labels)}")

    # Check for proper binary labels
    if not np.array_equal(np.sort(unique_labels), [-1, 1]):
        warnings.warn(f"Labels should be -1 and +1, got {unique_labels}")

    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")

    return True


if __name__ == "__main__":
    # Test the functions
    print("Testing data generators...")

    # Test linearly separable data
    print("\n1. Testing linearly separable data generation:")
    X_linear, y_linear = generate_linearly_separable_data(n_samples=100, random_state=42)
    validate_dataset(X_linear, y_linear)
    info_linear = get_dataset_info(X_linear, y_linear)
    print(f"   Samples: {info_linear['n_samples']}")
    print(f"   Classes: {info_linear['class_balance']}")
    print(
        f"   Feature ranges: X1=[{info_linear['feature_ranges']['min'][0]:.2f}, {info_linear['feature_ranges']['max'][0]:.2f}], X2=[{info_linear['feature_ranges']['min'][1]:.2f}, {info_linear['feature_ranges']['max'][1]:.2f}]")

    # Test non-linear data
    print("\n2. Testing non-linear data generation:")
    X_nonlinear, y_nonlinear = generate_nonlinear_data(n_samples=200, random_state=42)
    validate_dataset(X_nonlinear, y_nonlinear)
    info_nonlinear = get_dataset_info(X_nonlinear, y_nonlinear)
    print(f"   Samples: {info_nonlinear['n_samples']}")
    print(f"   Classes: {info_nonlinear['class_balance']}")
    print(
        f"   Feature ranges: X1=[{info_nonlinear['feature_ranges']['min'][0]:.2f}, {info_nonlinear['feature_ranges']['max'][0]:.2f}], X2=[{info_nonlinear['feature_ranges']['min'][1]:.2f}, {info_nonlinear['feature_ranges']['max'][1]:.2f}]")

    print("\n✅ Tests passed! Data generators working correctly.")
