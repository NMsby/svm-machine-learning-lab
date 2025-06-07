"""
Visualization Module for SVM Lab

This module provides comprehensive plotting functions for data visualization,
decision boundary plotting, and SVM analysis visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional, Tuple, Union, List
import warnings
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import os

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Define consistent color schemes
COLORS = {
    'class_1': '#e74c3c',  # Red for class +1
    'class_neg1': '#3498db',  # Blue for class -1
    'decision_boundary': '#2c3e50',  # Dark blue-gray
    'support_vectors': '#f39c12',  # Orange
    'margin': '#95a5a6',  # Gray
    'background_1': '#fadbd8',  # Light red
    'background_neg1': '#d6eaf8'  # Light blue
}

# Define markers
MARKERS = {
    'class_1': 'o',  # Circle for class +1
    'class_neg1': 's',  # Square for class -1
    'support_vectors': '^'  # Triangle for support vectors
}


def setup_plot_style(figsize: Tuple[int, int] = (10, 8),
                     style: str = 'whitegrid') -> None:
    """
    Set up consistent plotting style for all visualizations.

    Parameters:
    -----------
    figsize : tuple, default=(10, 8)
        Default figure size for plots
    style : str, default='whitegrid'
        Seaborn style to use
    """
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11

    sns.set_style(style)


def plot_2d_classification_data(
        X: np.ndarray,
        y: np.ndarray,
        title: str = "2D Classification Dataset",
        xlabel: str = "Feature 1",
        ylabel: str = "Feature 2",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        show_grid: bool = True,
        alpha: float = 0.7,
        s: int = 60
) -> plt.Figure:
    """
    Create a scatter plot for 2D binary classification data.

    Parameters:
    -----------
    X : np.ndarray of shape (n_samples, 2)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Binary labels (-1, +1)
    title : str, default="2D Classification Dataset"
        Plot title
    xlabel : str, default="Feature 1"
        X-axis label
    ylabel : str, default="Feature 2"
        Y-axis label
    figsize : tuple, default=(10, 8)
        Figure size
    save_path : str or None, default=None
        Path to save the plot. If None, the plot is not saved.
    show_grid : bool, default=True
        Whether to show grid
    alpha : float, default=0.7
        Transparency of points
    s : int, default=60
        Size of scatter points

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    # Input validation
    if X.shape[1] != 2:
        raise ValueError("X must have exactly 2 features for 2D plotting")

    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError("y must contain exactly 2 classes")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each class
    for label in unique_labels:
        mask = y == label
        color = COLORS['class_1'] if label > 0 else COLORS['class_neg1']
        marker = MARKERS['class_1'] if label > 0 else MARKERS['class_neg1']
        class_name = f'Class +1' if label > 0 else f'Class -1'

        ax.scatter(X[mask, 0], X[mask, 1],
                   c=color, marker=marker, s=s, alpha=alpha,
                   label=class_name, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend()

    if show_grid:
        ax.grid(True, alpha=0.3)

    # Make the plot look professional
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


def plot_decision_boundary(
        X: np.ndarray,
        y: np.ndarray,
        model,
        title: str = "SVM Decision Boundary",
        xlabel: str = "Feature 1",
        ylabel: str = "Feature 2",
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        show_support_vectors: bool = True,
        show_margins: bool = True,
        mesh_step: float = 0.02,
        alpha_boundary: float = 0.3,
        alpha_points: float = 0.8
) -> plt.Figure:
    """
    Plot decision boundary for SVM with optional support vectors and margins.

    Parameters:
    -----------
    X : np.ndarray of shape (n_samples, 2)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Binary labels
    model : object
        Trained SVM model with decision_function method
    title : str, default="SVM Decision Boundary"
        Plot title
    xlabel : str, default="Feature 1"
        X-axis label
    ylabel : str, default="Feature 2"
        Y-axis label
    figsize : tuple, default=(12, 10)
        Figure size
    save_path : str or None, default=None
        Path to save the plot
    show_support_vectors : bool, default=True
        Whether to highlight support vectors
    show_margins : bool, default=True
        Whether to show margin boundaries
    mesh_step : float, default=0.02
        Step size for mesh grid
    alpha_boundary : float, default=0.3
        Transparency for decision boundary regions
    alpha_points : float, default=0.8
        Transparency for data points

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Create mesh grid for decision boundary
    margin = 1.0
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))

    # Get decision function values for mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    try:
        # Try to get decision function values
        if hasattr(model, 'decision_function'):
            Z = model.decision_function(mesh_points)
        elif hasattr(model, 'predict'):
            Z = model.predict(mesh_points)
        else:
            raise AttributeError("Model must have decision_function or predict method")

        Z = Z.reshape(xx.shape)

        # Plot decision boundary and margins
        if show_margins and hasattr(model, 'decision_function'):
            # Plot decision boundary (Z=0) and margins (Z=±1)
            ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                       alpha=0.8, linestyles=['--', '-', '--'],
                       colors=[COLORS['margin'], COLORS['decision_boundary'], COLORS['margin']],
                       linewidths=[2, 3, 2])
        else:
            # Just plot decision boundary
            ax.contour(xx, yy, Z, levels=[0],
                       colors=[COLORS['decision_boundary']], linewidths=[3])

        # Fill regions with different colors
        ax.contourf(xx, yy, Z, levels=50, alpha=alpha_boundary,
                    cmap=ListedColormap([COLORS['background_neg1'], COLORS['background_1']]))

    except Exception as e:
        warnings.warn(f"Could not plot decision boundary: {e}")

    # Plot data points
    for label in np.unique(y):
        mask = y == label
        color = COLORS['class_1'] if label > 0 else COLORS['class_neg1']
        marker = MARKERS['class_1'] if label > 0 else MARKERS['class_neg1']
        class_name = f'Class +1' if label > 0 else f'Class -1'

        ax.scatter(X[mask, 0], X[mask, 1],
                   c=color, marker=marker, s=80, alpha=alpha_points,
                   label=class_name, edgecolors='black', linewidth=0.8)

    # Highlight support vectors if available and requested
    if show_support_vectors and hasattr(model, 'support_vectors_'):
        try:
            support_vectors = model.support_vectors_
            ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                       s=120, facecolors='none', edgecolors=COLORS['support_vectors'],
                       linewidth=3, label='Support Vectors', marker='o')
        except Exception as e:
            warnings.warn(f"Could not highlight support vectors: {e}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decision boundary plot saved to: {save_path}")

    return fig


def plot_training_history(
        costs: List[float],
        title: str = "SVM Training Cost History",
        xlabel: str = "Iteration",
        ylabel: str = "Cost",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
        show_convergence: bool = True
) -> plt.Figure:
    """
    Plot training cost history for SVM implementation.

    Parameters:
    -----------
    costs : list of float
        Training costs over iterations
    title : str, default="SVM Training Cost History"
        Plot title
    xlabel : str, default="Iteration"
        X-axis label
    ylabel : str, default="Cost"
        Y-axis label
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str or None, default=None
        Path to save the plot
    show_convergence : bool, default=True
        Whether to highlight convergence information

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    fig, ax = plt.subplots(figsize=figsize)

    iterations = range(1, len(costs) + 1)
    ax.plot(iterations, costs, linewidth=2, color=COLORS['decision_boundary'], alpha=0.8)

    # Add trend line for convergence analysis
    if show_convergence and len(costs) > 10:
        # Calculate moving average for trend
        window_size = max(5, len(costs) // 20)
        moving_avg = np.convolve(costs, np.ones(window_size) / window_size, mode='valid')
        moving_avg_x = range(window_size, len(costs) + 1)
        ax.plot(moving_avg_x, moving_avg, '--', linewidth=2,
                color=COLORS['class_1'], alpha=0.7, label='Moving Average')

        # Show final convergence value
        final_cost = costs[-1]
        ax.axhline(y=final_cost, color=COLORS['support_vectors'],
                   linestyle=':', alpha=0.7, label=f'Final Cost: {final_cost:.4f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if show_convergence and len(costs) > 10:
        ax.legend()

    # Make the plot look professional
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    return fig


def plot_multiple_kernels_comparison(
        X: np.ndarray,
        y: np.ndarray,
        models_dict: dict,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare decision boundaries for multiple kernel types.

    Parameters:
    -----------
    X : np.ndarray of shape (n_samples, 2)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Binary labels
    models_dict : dict
        Dictionary with kernel names as keys and trained models as values
    figsize : tuple, default=(15, 10)
        Figure size
    save_path : str or None, default=None
        Path to save the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    n_models = len(models_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (kernel_name, model) in enumerate(models_dict.items()):
        ax = axes[idx]

        # Create mesh grid
        margin = 1.0
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        # Get decision function values
        try:
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            if hasattr(model, 'decision_function'):
                Z = model.decision_function(mesh_points)
            else:
                Z = model.predict(mesh_points)
            Z = Z.reshape(xx.shape)

            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.3,
                        cmap=ListedColormap([COLORS['background_neg1'], COLORS['background_1']]))
            ax.contour(xx, yy, Z, levels=[0], colors=[COLORS['decision_boundary']], linewidths=[2])

        except Exception as e:
            warnings.warn(f"Could not plot decision boundary for {kernel_name}: {e}")

        # Plot data points
        for label in np.unique(y):
            mask = y == label
            color = COLORS['class_1'] if label > 0 else COLORS['class_neg1']
            marker = MARKERS['class_1'] if label > 0 else MARKERS['class_neg1']
            ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, s=50, alpha=0.8,
                       edgecolors='black', linewidth=0.5)

        # Highlight support vectors if available
        if hasattr(model, 'support_vectors_'):
            try:
                support_vectors = model.support_vectors_
                ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                           s=100, facecolors='none', edgecolors=COLORS['support_vectors'],
                           linewidth=2, marker='o')
            except:
                pass

        ax.set_title(f'{kernel_name.capitalize()} Kernel', fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kernel comparison plot saved to: {save_path}")

    return fig


def plot_hyperparameter_analysis(
        results_df: pd.DataFrame,
        param_name: str,
        metric: str = 'val_accuracy',
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot hyperparameter analysis results.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from hyperparameter grid search
    param_name : str
        Parameter name to analyze
    metric : str, default='val_accuracy'
        Metric to plot
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str or None, default=None
        Path to save the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Group by parameter value
    grouped = results_df.groupby(param_name)[metric]

    # Box plot
    param_values = []
    metric_values = []
    for param_val, group in grouped:
        param_values.extend([param_val] * len(group))
        metric_values.extend(group.values)

    axes[0].boxplot([grouped.get_group(val).values for val in sorted(grouped.groups.keys())],
                    labels=sorted(grouped.groups.keys()))
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel(metric.replace('_', ' ').title())
    axes[0].set_title(f'{metric.replace("_", " ").title()} vs {param_name}')
    axes[0].grid(True, alpha=0.3)

    # Line plot with error bars
    param_means = grouped.mean()
    param_stds = grouped.std()

    axes[1].errorbar(param_means.index, param_means.values, yerr=param_stds.values,
                     marker='o', capsize=5, capthick=2, linewidth=2)
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel(metric.replace('_', ' ').title())
    axes[1].set_title(f'Mean {metric.replace("_", " ").title()} vs {param_name}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hyperparameter analysis plot saved to: {save_path}")

    return fig


def plot_learning_curve(
        learning_curve_data: dict,
        title: str = "Learning Curve",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning curve showing training and validation performance vs dataset size.

    Parameters:
    -----------
    learning_curve_data : dict
        Data from analyze_learning_curve function
    title : str, default="Learning Curve"
        Plot title
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str or None, default=None
        Path to save the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    train_sizes = learning_curve_data['n_samples_used']
    train_scores = learning_curve_data['train_scores']
    val_scores = learning_curve_data['val_scores']
    train_times = learning_curve_data['train_times']

    # Learning curve plot
    axes[0].plot(train_sizes, train_scores, 'o-', color=COLORS['class_1'],
                 linewidth=2, markersize=6, label='Training Score')
    axes[0].plot(train_sizes, val_scores, 'o-', color=COLORS['class_neg1'],
                 linewidth=2, markersize=6, label='Validation Score')

    axes[0].set_xlabel('Training Set Size')
    axes[0].set_ylabel('Accuracy Score')
    axes[0].set_title(f'{title} - Performance vs Dataset Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Training time plot
    axes[1].plot(train_sizes, train_times, 'o-', color=COLORS['support_vectors'],
                 linewidth=2, markersize=6)
    axes[1].set_xlabel('Training Set Size')
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title(f'{title} - Training Time vs Dataset Size')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve plot saved to: {save_path}")

    return fig


def plot_cross_validation_results(
        cv_results: dict,
        title: str = "Cross-Validation Results",
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cross-validation results with detailed metrics.

    Parameters:
    -----------
    cv_results : dict
        Results from cross_validate_model function
    title : str, default="Cross-Validation Results"
        Plot title
    figsize : tuple, default=(14, 6)
        Figure size
    save_path : str or None, default=None
        Path to save the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Box plot of all metrics
    metric_values = []
    for metric in metrics:
        metric_values.append(cv_results[f'{metric}_values'])

    box_plot = axes[0].boxplot(metric_values, labels=metric_names, patch_artist=True)

    # Color the boxes
    colors = [COLORS['class_1'], COLORS['class_neg1'], COLORS['support_vectors'], COLORS['decision_boundary']]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0].set_ylabel('Score')
    axes[0].set_title('Cross-Validation Metrics Distribution')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Fold-by-fold performance
    fold_numbers = range(1, len(cv_results['accuracy_values']) + 1)
    axes[1].plot(fold_numbers, cv_results['accuracy_values'], 'o-',
                 color=COLORS['class_1'], linewidth=2, markersize=8, label='Accuracy')
    axes[1].plot(fold_numbers, cv_results['f1_score_values'], 'o-',
                 color=COLORS['class_neg1'], linewidth=2, markersize=8, label='F1 Score')

    axes[1].set_xlabel('Fold Number')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Performance Across Folds')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    # Mean ± std visualization
    metrics_means = [cv_results[f'{metric}_mean'] for metric in metrics]
    metrics_stds = [cv_results[f'{metric}_std'] for metric in metrics]

    x_pos = np.arange(len(metrics))
    bars = axes[2].bar(x_pos, metrics_means, yerr=metrics_stds, capsize=5,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(metric_names)
    axes[2].set_ylabel('Score')
    axes[2].set_title('Mean Performance ± Standard Deviation')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim(0, 1.05)

    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, metrics_means, metrics_stds):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2., height + std_val + 0.02,
                     f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-validation results plot saved to: {save_path}")

    return fig


def plot_confusion_matrix_comparison(
        models_data: List[Tuple[str, np.ndarray, np.ndarray]],
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrices for multiple models side by side.

    Parameters:
    -----------
    models_data : list of tuples
        List of (model_name, y_true, y_pred) tuples
    figsize : tuple, default=(15, 5)
        Figure size
    save_path : str or None, default=None
        Path to save the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """

    n_models = len(models_data)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, y_true, y_pred) in enumerate(models_data):
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])

        # Normalize to percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot
        im = axes[idx].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        axes[idx].set_title(f'{model_name}\nConfusion Matrix')

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label('Percentage (%)')

        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)',
                               ha="center", va="center",
                               color="white" if cm_normalized[i, j] > thresh else "black",
                               fontsize=10, fontweight='bold')

        # Set labels
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Class -1', 'Class +1'])
        axes[idx].set_yticklabels(['Class -1', 'Class +1'])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix comparison saved to: {save_path}")

    return fig


def save_all_plots(save_dir: str = "results/plots") -> None:
    """
    Create directory structure for saving plots.

    Parameters:
    -----------
    save_dir : str, default="results/plots"
        Base directory for saving plots
    """
    subdirs = ['data_exploration', 'decision_boundaries', 'training_history',
               'kernel_comparison', 'performance_analysis']

    for subdir in subdirs:
        full_path = os.path.join(save_dir, subdir)
        os.makedirs(full_path, exist_ok=True)

    print(f"Plot directories created in: {save_dir}")


# Initialize plotting style when module is imported
setup_plot_style()

if __name__ == "__main__":
    # Test the visualization functions
    print("Testing visualization module...")

    # Import data generators for testing
    import sys

    sys.path.append('.')
    from data_generators import generate_linearly_separable_data, generate_nonlinear_data

    # Generate test data
    X_linear, y_linear = generate_linearly_separable_data(n_samples=100, random_state=42)
    X_nonlinear, y_nonlinear = generate_nonlinear_data(n_samples=200, random_state=42)

    # Test basic plotting
    print("\n1. Testing basic data visualization:")
    fig1 = plot_2d_classification_data(X_linear, y_linear,
                                       title="Test Linear Data")
    plt.show()

    fig2 = plot_2d_classification_data(X_nonlinear, y_nonlinear,
                                       title="Test Non-linear Data")
    plt.show()

    # Test training history plotting
    print("\n2. Testing training history visualization:")
    # Simulate training costs
    test_costs = [10.0 * np.exp(-0.1 * i) + 0.5 + 0.1 * np.random.randn() for i in range(100)]
    fig3 = plot_training_history(test_costs, title="Test Training History")
    plt.show()

    # Create plot directories
    print("\n3. Creating plot directory structure:")
    save_all_plots()

    print("\n✅ All visualization tests passed! Module is working correctly.")
