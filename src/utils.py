"""
Utility Functions for SVM Lab

This module provides comprehensive utilities for performance evaluation,
hyperparameter tuning, model comparison, and statistical analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import time
import warnings
from itertools import product


def calculate_comprehensive_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        model_name: str = "Model"
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for binary classification.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities (for probability-based metrics)
    model_name : str, default="Model"
        Name of the model for display

    Returns:
    --------
    metrics : dict
        Dictionary containing all performance metrics
    """

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel()

    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall

    # Balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2

    # Matthews Correlation Coefficient
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'balanced_accuracy': balanced_accuracy,
        'mcc': mcc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': len(y_true)
    }

    return metrics


def compare_models_performance(
        models_results: List[Dict[str, Any]],
        save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare performance of multiple models.

    Parameters:
    -----------
    models_results : list of dict
        List of model results from calculate_comprehensive_metrics
    save_path : str, optional
        Path to save comparison table

    Returns:
    --------
    comparison_df : pd.DataFrame
        DataFrame with model comparison
    """

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(models_results)

    # Select key metrics for comparison
    key_metrics = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score',
                   'balanced_accuracy', 'mcc']

    if all(metric in comparison_df.columns for metric in key_metrics):
        display_df = comparison_df[key_metrics].round(4)
    else:
        display_df = comparison_df

    # Sort by accuracy (descending)
    display_df = display_df.sort_values('accuracy', ascending=False)

    # Add ranking
    display_df['rank'] = range(1, len(display_df) + 1)

    # Reorder columns to put rank first
    cols = ['rank'] + [col for col in display_df.columns if col != 'rank']
    display_df = display_df[cols]

    if save_path:
        display_df.to_csv(save_path, index=False)
        print(f"Model comparison saved to: {save_path}")

    return display_df


def hyperparameter_grid_search(
        model_class,
        param_grid: Dict[str, List],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scoring_metric: str = 'accuracy',
        verbose: bool = True
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Perform grid search for hyperparameter tuning.

    Parameters:
    -----------
    model_class : class
        Model class to instantiate
    param_grid : dict
        Dictionary with parameter names as keys and lists of values to try
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    scoring_metric : str, default='accuracy'
        Metric to optimize ('accuracy', 'f1', 'precision', 'recall')
    verbose : bool, default=True
        Whether to print progress

    Returns:
    --------
    best_params : dict
        Best parameter combination
    best_score : float
        Best validation score
    results_df : pd.DataFrame
        DataFrame with all results
    """

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    results = []
    best_score = -np.inf
    best_params = None

    if verbose:
        print(f"Starting grid search with {len(param_combinations)} combinations...")
        print(f"Optimizing for: {scoring_metric}")
        print("-" * 60)

    for i, param_combo in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, param_combo))

        try:
            # Create and train model
            start_time = time.time()
            model = model_class(**params)
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Make predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate metrics
            train_metrics = calculate_comprehensive_metrics(y_train, train_pred, model_name="train")
            val_metrics = calculate_comprehensive_metrics(y_val, val_pred, model_name="val")

            # Get score for optimization
            if scoring_metric in val_metrics:
                score = val_metrics[scoring_metric]
            else:
                score = val_metrics['accuracy']  # fallback

            # Store results
            result = {
                'params': params,
                'train_accuracy': train_metrics['accuracy'],
                'val_accuracy': val_metrics['accuracy'],
                'train_f1': train_metrics['f1_score'],
                'val_f1': val_metrics['f1_score'],
                'score': score,
                'training_time': training_time,
                'param_string': str(params)
            }

            # Add individual parameters for easier analysis
            for param_name, param_value in params.items():
                result[param_name] = param_value

            results.append(result)

            # Update best parameters
            if score > best_score:
                best_score = score
                best_params = params.copy()

            if verbose and (i + 1) % max(1, len(param_combinations) // 10) == 0:
                print(f"Progress: {i + 1:3d}/{len(param_combinations)} | "
                      f"Current best {scoring_metric}: {best_score:.4f} | "
                      f"Current: {score:.4f}")

        except Exception as e:
            if verbose:
                print(f"Error with params {params}: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('score', ascending=False)

    if verbose:
        print("-" * 60)
        print(f"Grid search completed!")
        print(f"Best {scoring_metric}: {best_score:.4f}")
        print(f"Best parameters: {best_params}")

    return best_params, best_score, results_df


def cross_validate_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform cross-validation evaluation of a model.

    Parameters:
    -----------
    model : object
        Trained model with fit and predict methods
    X, y : np.ndarray
        Data for cross-validation
    cv_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print results

    Returns:
    --------
    cv_results : dict
        Cross-validation results
    """

    from sklearn.model_selection import StratifiedKFold

    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_results = []
    fold_times = []

    if verbose:
        print(f"Performing {cv_folds}-fold cross-validation...")
        print("-" * 40)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Train model
        start_time = time.time()
        model.fit(X_train_fold, y_train_fold)
        training_time = time.time() - start_time
        fold_times.append(training_time)

        # Predict
        val_pred = model.predict(X_val_fold)

        # Calculate metrics
        fold_metrics = calculate_comprehensive_metrics(y_val_fold, val_pred,
                                                       model_name=f"fold_{fold + 1}")
        fold_metrics['training_time'] = training_time
        fold_results.append(fold_metrics)

        if verbose:
            print(f"Fold {fold + 1}: Accuracy = {fold_metrics['accuracy']:.4f}, "
                  f"F1 = {fold_metrics['f1_score']:.4f}")

    # Calculate statistics
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']
    cv_stats = {}

    for metric in metrics_names:
        values = [result[metric] for result in fold_results]
        cv_stats[f'{metric}_mean'] = np.mean(values)
        cv_stats[f'{metric}_std'] = np.std(values)
        cv_stats[f'{metric}_values'] = values

    cv_stats['training_time_mean'] = np.mean(fold_times)
    cv_stats['training_time_std'] = np.std(fold_times)
    cv_stats['fold_results'] = fold_results

    if verbose:
        print("-" * 40)
        print(f"Cross-validation Results:")
        print(f"Accuracy: {cv_stats['accuracy_mean']:.4f} ± {cv_stats['accuracy_std']:.4f}")
        print(f"F1 Score: {cv_stats['f1_score_mean']:.4f} ± {cv_stats['f1_score_std']:.4f}")
        print(f"Training time: {cv_stats['training_time_mean']:.3f}s ± {cv_stats['training_time_std']:.3f}s")

    return cv_stats


def benchmark_training_time(
        model_class,
        params: Dict,
        X: np.ndarray,
        y: np.ndarray,
        n_runs: int = 5,
        verbose: bool = True
) -> Dict[str, float]:
    """
    Benchmark training time for a model with multiple runs.

    Parameters:
    -----------
    model_class : class
        Model class to benchmark
    params : dict
        Model parameters
    X, y : np.ndarray
        Training data
    n_runs : int, default=5
        Number of runs for averaging
    verbose : bool, default=True
        Whether to print results

    Returns:
    --------
    timing_results : dict
        Timing statistics
    """

    times = []
    accuracies = []

    if verbose:
        print(f"Benchmarking training time over {n_runs} runs...")

    for run in range(n_runs):
        # Create fresh model instance
        model = model_class(**params)

        # Time the training
        start_time = time.time()
        model.fit(X, y)
        end_time = time.time()

        training_time = end_time - start_time
        times.append(training_time)

        # Check consistency by measuring accuracy
        pred = model.predict(X)
        accuracy = accuracy_score(y, pred)
        accuracies.append(accuracy)

        if verbose:
            print(f"Run {run + 1}: {training_time:.4f}s, Accuracy: {accuracy:.4f}")

    results = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'all_times': times,
        'all_accuracies': accuracies
    }

    if verbose:
        print(f"\nBenchmark Results:")
        print(f"Training time: {results['mean_time']:.4f}s ± {results['std_time']:.4f}s")
        print(f"Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")

    return results


def analyze_learning_curve(
        model,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: Optional[np.ndarray] = None,
        cv_folds: int = 3,
        random_state: int = 42,
        verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Analyze learning curve by training on different dataset sizes.

    Parameters:
    -----------
    model : object
        Model instance
    X, y : np.ndarray
        Full dataset
    train_sizes : np.ndarray, optional
        Fractions of dataset to use. If None, uses default range.
    cv_folds : int, default=3
        Number of cross-validation folds
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print progress

    Returns:
    --------
    learning_curve_results : dict
        Learning curve data
    """

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    n_samples = len(X)
    train_scores = []
    val_scores = []
    train_times = []

    if verbose:
        print("Analyzing learning curve...")
        print("-" * 40)

    for train_size in train_sizes:
        n_train = int(n_samples * train_size)

        # Perform cross-validation for this training size
        fold_train_scores = []
        fold_val_scores = []
        fold_times = []

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        for train_idx, val_idx in skf.split(X, y):
            # Use only subset of training data
            train_subset_idx = train_idx[:n_train]

            X_train_fold = X[train_subset_idx]
            y_train_fold = y[train_subset_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            # Train model
            start_time = time.time()
            model.fit(X_train_fold, y_train_fold)
            training_time = time.time() - start_time

            # Evaluate
            train_pred = model.predict(X_train_fold)
            val_pred = model.predict(X_val_fold)

            train_acc = accuracy_score(y_train_fold, train_pred)
            val_acc = accuracy_score(y_val_fold, val_pred)

            fold_train_scores.append(train_acc)
            fold_val_scores.append(val_acc)
            fold_times.append(training_time)

        # Average across folds
        train_scores.append(np.mean(fold_train_scores))
        val_scores.append(np.mean(fold_val_scores))
        train_times.append(np.mean(fold_times))

        if verbose:
            print(f"Size: {train_size:.1f} ({n_train:3d} samples) | "
                  f"Train: {np.mean(fold_train_scores):.4f} | "
                  f"Val: {np.mean(fold_val_scores):.4f}")

    results = {
        'train_sizes': train_sizes,
        'train_scores': np.array(train_scores),
        'val_scores': np.array(val_scores),
        'train_times': np.array(train_times),
        'n_samples_used': (train_sizes * n_samples).astype(int)
    }

    return results


def statistical_significance_test(
        results1: Dict[str, List[float]],
        results2: Dict[str, List[float]],
        metric: str = 'accuracy',
        alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test statistical significance between two sets of cross-validation results.

    Parameters:
    -----------
    results1, results2 : dict
        Cross-validation results from cross_validate_model
    metric : str, default='accuracy'
        Metric to test
    alpha : float, default=0.05
        Significance level

    Returns:
    --------
    test_results : dict
        Statistical test results
    """

    from scipy import stats

    values1 = results1[f'{metric}_values']
    values2 = results2[f'{metric}_values']

    # Paired t-test (since we use same CV folds)
    t_stat, p_value = stats.ttest_rel(values1, values2)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((np.var(values1) + np.var(values2)) / 2))
    cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std

    significant = p_value < alpha

    results = {
        'metric': metric,
        'mean_diff': np.mean(values1) - np.mean(values2),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': significant,
        'alpha': alpha,
        'values1': values1,
        'values2': values2
    }

    return results


if __name__ == "__main__":
    # Test the utility functions
    print("Testing utility functions...")
    print("=" * 50)

    # Generate test data
    np.random.seed(42)
    n_samples = 100
    X_test = np.random.randn(n_samples, 2)
    y_test = np.random.choice([-1, 1], n_samples)
    y_pred_test = np.random.choice([-1, 1], n_samples)

    # Test comprehensive metrics
    print("\n1. Testing comprehensive metrics calculation:")
    metrics = calculate_comprehensive_metrics(y_test, y_pred_test, model_name="Test Model")
    print(f"Calculated {len(metrics)} metrics for test data")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    # Test model comparison
    print("\n2. Testing model comparison:")
    models_results = [
        metrics,
        calculate_comprehensive_metrics(y_test, y_pred_test, model_name="Test Model 2")
    ]
    comparison_df = compare_models_performance(models_results)
    print(f"Comparison table shape: {comparison_df.shape}")

    print("\nAll utility function tests passed!")
