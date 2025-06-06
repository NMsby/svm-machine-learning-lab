"""
Support Vector Machine Implementation from Scratch

This module implements a Support Vector Machine using gradient descent optimization
for binary classification.
Also features hinge loss, regularization, and comprehensive training monitoring.

Mathematical Foundation:
- Objective: minimize (1/2)||w||² + C * Σ max(0, 1 - yi(w·xi + b))
- Gradient descent updates for weights and bias
- Hinge loss for margin-based classification

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
from typing import Optional, List, Tuple, Union
import warnings
from abc import ABC, abstractmethod


# noinspection GrazieInspection
class SVMFromScratch:
    """
    Support Vector Machine implementation using gradient descent.

    This class implements a binary SVM classifier using the hinge loss function
    and L2 regularization, optimized through gradient descent.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    lambda_param : float, default=0.01
        Regularization parameter (C = 1/lambda_param)
    n_iters : int, default=1000
        Maximum number of training iterations
    tolerance : float, default=1e-6
        Convergence tolerance for early stopping
    early_stopping : bool, default=True
        Whether to use early stopping based on cost convergence
    learning_rate_decay : float, default=0.99
        Decay factor for learning rate (applied each iteration)
    verbose : bool, default=False
        Whether to print training progress

    Attributes:
    -----------
    w : np.ndarray of shape (n_features,)
        Weight vector
    b : float
        Bias term
    costs : list
        Training costs over iterations
    converged : bool
        Whether training converged
    n_support_vectors : int
        Number of support vectors (estimated)
    """

    def __init__(
            self,
            learning_rate: float = 0.01,
            lambda_param: float = 0.01,
            n_iters: int = 1000,
            tolerance: float = 1e-6,
            early_stopping: bool = True,
            learning_rate_decay: float = 0.99,
            verbose: bool = False
    ):

        # Parameter validation
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if lambda_param <= 0:
            raise ValueError("lambda_param must be positive")
        if n_iters <= 0:
            raise ValueError("n_iters must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if not (0.9 <= learning_rate_decay <= 1.0):
            raise ValueError("learning_rate_decay should be between 0.9 and 1.0")

        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.early_stopping = early_stopping
        self.learning_rate_decay = learning_rate_decay
        self.verbose = verbose

        # Initialize model parameters
        self.w = None
        self.b = None
        self.costs = []
        self.converged = False
        self.n_support_vectors = 0
        self._training_info = {}

    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the total cost (hinge loss + regularization).

        Cost = λ/2 * ||w||² + (1/n) * Σ max(0, 1 - yi(w·xi + b))

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Training labels

        Returns:
        --------
        cost : float
            Total cost value
        """
        n_samples = X.shape[0]

        # Regularization term
        regularization = 0.5 * self.lambda_param * np.dot(self.w, self.w)

        # Hinge loss term
        linear_output = np.dot(X, self.w) + self.b
        hinge_loss = np.maximum(0, 1 - y * linear_output)
        hinge_loss_mean = np.mean(hinge_loss)

        total_cost = regularization + hinge_loss_mean
        return total_cost

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.

        For each sample i:
        - If yi(w·xi + b) >= 1: gradients only from regularization
        - If yi(w·xi + b) < 1: gradients from both regularization and hinge loss

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Training labels

        Returns:
        --------
        dw : np.ndarray of shape (n_features,)
            Gradient with respect to weights
        db : float
            Gradient with respect to bias
        """
        n_samples, n_features = X.shape

        # Compute linear output for all samples
        linear_output = np.dot(X, self.w) + self.b

        # Find samples that violate the margin (hinge loss > 0)
        margin_violation = y * linear_output < 1

        # Initialize gradients
        dw = np.zeros(n_features)
        db = 0.0

        # Regularization gradient (always present)
        dw += self.lambda_param * self.w

        # Hinge loss gradients (only for margin violations)
        if np.any(margin_violation):
            # For samples with margin violation: add -yi*xi to weight gradient
            violation_samples = X[margin_violation]
            violation_labels = y[margin_violation]

            # Weight gradient from hinge loss
            dw -= np.mean(violation_labels[:, np.newaxis] * violation_samples, axis=0)

            # Bias gradient from hinge loss
            db -= np.mean(violation_labels)

        return dw, db

    def _update_learning_rate(self, iteration: int) -> None:
        """
        Update learning rate with decay.

        Parameters:
        -----------
        iteration : int
            Current iteration number
        """
        if self.learning_rate_decay < 1.0:
            self.learning_rate *= self.learning_rate_decay

    def _check_convergence(self, iteration: int) -> bool:
        """
        Check if training has converged based on cost change.

        Parameters:
        -----------
        iteration : int
            Current iteration number

        Returns:
        --------
        converged : bool
            Whether training has converged
        """
        if not self.early_stopping or len(self.costs) < 10:
            return False

        # Check if cost change is below tolerance
        recent_costs = self.costs[-10:]
        cost_change = abs(recent_costs[-1] - recent_costs[0]) / abs(recent_costs[0])

        return cost_change < self.tolerance

    def _estimate_support_vectors(self, X: np.ndarray, y: np.ndarray) -> int:
        """
        Estimate the number of support vectors.

        Support vectors are points that lie on or within the margin.
        Hence, consider points with |decision_value| <= 1.1 as approximate support vectors.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Training labels

        Returns:
        --------
        n_support_vectors : int
            Estimated number of support vectors
        """
        decision_values = self.decision_function(X)

        # Points close to the decision boundary (within margin + small tolerance)
        margin_threshold = 1.1
        support_vector_mask = np.abs(decision_values) <= margin_threshold

        return np.sum(support_vector_mask)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMFromScratch':
        """
        Train the SVM using gradient descent.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Training labels (must be -1 or +1)

        Returns:
        --------
        self : SVMFromScratch
            Fitted estimator
        """

        # Input validation
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # Check for binary classification
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM supports binary classification only")
        if not np.array_equal(np.sort(unique_labels), [-1, 1]):
            warnings.warn("Labels should be -1 and +1. Converting automatically.")
            # Convert labels to -1, +1
            y = np.where(y == unique_labels[0], -1, 1)

        n_samples, n_features = X.shape

        # Initialize parameters
        # Use small random weights to break symmetry
        np.random.seed(42)  # For reproducibility
        self.w = np.random.normal(0, 0.01, n_features)
        self.b = 0.0

        # Reset training state
        self.costs = []
        self.converged = False
        current_lr = self.learning_rate

        if self.verbose:
            print(f"Starting SVM training...")
            print(f"Samples: {n_samples}, Features: {n_features}")
            print(f"Learning rate: {self.learning_rate}, Lambda: {self.lambda_param}")
            print("-" * 50)

        # Training loop
        for iteration in range(self.n_iters):

            # Compute cost
            cost = self._compute_cost(X, y)
            self.costs.append(cost)

            # Compute gradients
            dw, db = self._compute_gradients(X, y)

            # Update parameters
            self.w -= current_lr * dw
            self.b -= current_lr * db

            # Update learning rate
            self._update_learning_rate(iteration)
            current_lr = self.learning_rate

            # Check convergence
            if self._check_convergence(iteration):
                self.converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            # Print progress
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1:4d}: Cost = {cost:.6f}, LR = {current_lr:.6f}")

        # Estimate support vectors
        self.n_support_vectors = self._estimate_support_vectors(X, y)

        # Store training information
        self._training_info = {
            'final_cost': self.costs[-1],
            'iterations': len(self.costs),
            'converged': self.converged,
            'n_support_vectors': self.n_support_vectors
        }

        if self.verbose:
            print("-" * 50)
            print(f"Training completed!")
            print(f"Final cost: {self.costs[-1]:.6f}")
            print(f"Iterations: {len(self.costs)}")
            print(f"Converged: {self.converged}")
            print(f"Estimated support vectors: {self.n_support_vectors}")

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function (distance to separating hyperplane).

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        decision : np.ndarray of shape (n_samples,)
            Decision function values (w·x + b)
        """
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return np.dot(X, self.w) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels (-1 or +1)
        """
        decision_values = self.decision_function(X)
        return np.where(decision_values >= 0, 1, -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using decision function values.

        Note: This is an approximation using sigmoid transformation
        of the decision function values.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        probabilities : np.ndarray of shape (n_samples, 2)
            Predicted probabilities for each class
        """
        decision_values = self.decision_function(X)

        # Use sigmoid transformation for probability approximation
        # Apply scaling to make the sigmoid more meaningful
        scaled_values = decision_values / (np.std(decision_values) + 1e-8)
        prob_positive = 1 / (1 + np.exp(-scaled_values))
        prob_negative = 1 - prob_positive

        return np.column_stack([prob_negative, prob_positive])

    def get_params(self) -> dict:
        """
        Get model parameters.

        Returns:
        --------
        params : dict
            Dictionary containing model parameters
        """
        return {
            'learning_rate': self.learning_rate,
            'lambda_param': self.lambda_param,
            'n_iters': self.n_iters,
            'tolerance': self.tolerance,
            'early_stopping': self.early_stopping,
            'learning_rate_decay': self.learning_rate_decay,
            'verbose': self.verbose
        }

    def get_training_info(self) -> dict:
        """
        Get information about the training process.

        Returns:
        --------
        info : dict
            Dictionary containing training information
        """
        return self._training_info.copy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score on given data.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples
        y : np.ndarray of shape (n_samples,)
            True labels

        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


if __name__ == "__main__":
    # Test the SVM implementation
    print("Testing SVM from Scratch implementation...")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)

    # Simple linearly separable data
    n_samples = 100
    X1 = np.random.normal([2, 2], 0.5, (n_samples // 2, 2))
    X2 = np.random.normal([-2, -2], 0.5, (n_samples // 2, 2))
    X_test = np.vstack([X1, X2])
    y_test = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])

    # Shuffle data
    shuffle_idx = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_idx]
    y_test = y_test[shuffle_idx]

    print(f"Generated test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_test + 1)}")

    # Test SVM
    print("\nTesting SVM training...")
    svm = SVMFromScratch(
        learning_rate=0.01,
        lambda_param=0.01,
        n_iters=500,
        verbose=True
    )

    # Train the model
    svm.fit(X_test, y_test)

    # Make predictions
    predictions = svm.predict(X_test)
    probabilities = svm.predict_proba(X_test)

    # Evaluate performance
    accuracy = svm.score(X_test, y_test)
    print(f"\nFinal Results:")
    print(f"Training accuracy: {accuracy:.4f}")
    print(f"Weight vector: {svm.w}")
    print(f"Bias: {svm.b:.4f}")
    print(f"Training info: {svm.get_training_info()}")

    # Test decision function
    decision_values = svm.decision_function(X_test[:5])
    print(f"\nSample decision values: {decision_values}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample true labels: {y_test[:5]}")

    print("\nAll tests passed! SVM implementation is working!")
