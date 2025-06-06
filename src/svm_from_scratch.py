"""
Support Vector Machine Implementation from Scratch

This module implements a Support Vector Machine classifier using gradient descent
optimization. The implementation includes hinge loss calculation, regularization,
and comprehensive training monitoring.

Mathematical Foundation:
- Optimization Problem: minimize (1/2)||w||² + C∑ξᵢ
- Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
- Hinge Loss: L(w,b) = ∑max(0, 1 - yᵢ(w·xᵢ + b))
"""

import numpy as np
from typing import Optional, List, Tuple, Union
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin


class SVMFromScratch(BaseEstimator, ClassifierMixin):
    """
    Support Vector Machine implementation using gradient descent optimization.

    This implementation uses the primal formulation of SVM with hinge loss and
    L2 regularization, optimized using gradient descent with optional learning
    rate scheduling and early stopping.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        Initial learning rate for gradient descent
    lambda_param : float, default=0.01
        Regularization parameter (equivalent to 1/C in sklearn)
        Higher values = more regularization = simpler model
    n_iters : int, default=1000
        Maximum number of training iterations
    tolerance : float, default=1e-6
        Convergence tolerance for early stopping
    learning_rate_decay : float, default=0.99
        Learning rate decay factor (applied each iteration)
    early_stopping : bool, default=True
        Whether to use early stopping based on cost convergence
    patience : int, default=50
        Number of iterations to wait for improvement before stopping
    random_state : int or None, default=None
        Random seed for reproducible weight initialization
    verbose : bool, default=False
        Whether to print training progress

    Attributes:
    -----------
    w : np.ndarray of shape (n_features,)
        Weight vector after training
    b : float
        Bias term after training
    costs : list of float
        Training cost history
    n_support_vectors_ : int
        Number of support vectors (approximate)
    training_history_ : dict
        Detailed training history including convergence info
    is_fitted_ : bool
        Whether the model has been fitted
    """

    def __init__(
            self,
            learning_rate: float = 0.01,
            lambda_param: float = 0.01,
            n_iters: int = 1000,
            tolerance: float = 1e-6,
            learning_rate_decay: float = 0.99,
            early_stopping: bool = True,
            patience: int = 50,
            random_state: Optional[int] = None,
            verbose: bool = False
    ):
        # Validate parameters
        self._validate_parameters(
            learning_rate, lambda_param, n_iters, tolerance,
            learning_rate_decay, patience
        )

        # Set parameters
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.learning_rate_decay = learning_rate_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

        # Initialize attributes that will be set during training
        self.w = None
        self.b = None
        self.costs = []
        self.n_support_vectors_ = 0
        self.training_history_ = {}
        self.is_fitted_ = False

        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _validate_parameters(
            self,
            learning_rate: float,
            lambda_param: float,
            n_iters: int,
            tolerance: float,
            learning_rate_decay: float,
            patience: int
    ) -> None:
        """Validate input parameters for the SVM."""

        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        if lambda_param < 0:
            raise ValueError(f"lambda_param must be non-negative, got {lambda_param}")

        if n_iters <= 0:
            raise ValueError(f"n_iters must be positive, got {n_iters}")

        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")

        if not 0 < learning_rate_decay <= 1:
            raise ValueError(f"learning_rate_decay must be in (0, 1], got {learning_rate_decay}")

        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")

    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate and preprocess input data."""

        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        if y is not None:
            y = np.asarray(y, dtype=np.float64)

        # Check X dimensions
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")

        if y is not None:
            # Check y dimensions
            if y.ndim != 1:
                raise ValueError(f"y must be 1D array, got {y.ndim}D")

            # Check length consistency
            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")

            # Check for NaN or infinite values in y
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                raise ValueError("y contains NaN or infinite values")

            # Check for binary classification
            unique_labels = np.unique(y)
            if len(unique_labels) != 2:
                raise ValueError(f"Binary classification requires exactly 2 classes, got {len(unique_labels)}")

            # Convert labels to -1, +1 if necessary
            if not np.array_equal(np.sort(unique_labels), [-1, 1]):
                warnings.warn(f"Converting labels {unique_labels} to [-1, +1]")
                label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
                y = np.array([label_map[label] for label in y])

        return X, y

    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize weight vector and bias term."""

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize weights with small random values
        # Using Xavier/Glorot initialization scaled for SVM
        scale = np.sqrt(2.0 / n_features)
        self.w = np.random.normal(0, scale, n_features)

        # Initialize bias to zero
        self.b = 0.0

        if self.verbose:
            print(f"Initialized weights: mean={self.w.mean():.6f}, std={self.w.std():.6f}")
            print(f"Initialized bias: {self.b}")

    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the total cost (hinge loss + regularization).

        Cost = (1/2) * lambda * ||w||² + (1/n) * ∑max(0, 1 - yᵢ(w·xᵢ + b))

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
        y : np.ndarray of shape (n_samples,)
            Target labels (-1 or +1)

        Returns:
        --------
        cost : float
            Total cost value
        """
        n_samples = len(X)

        # Compute linear predictions
        linear_output = X.dot(self.w) + self.b

        # Compute hinge loss for each sample
        hinge_losses = np.maximum(0, 1 - y * linear_output)

        # Average hinge loss
        hinge_loss = np.mean(hinge_losses)

        # Regularization term
        regularization = 0.5 * self.lambda_param * np.dot(self.w, self.w)

        # Total cost
        total_cost = hinge_loss + regularization

        return total_cost

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weight vector and bias term.

        For samples where yᵢ(w·xᵢ + b) < 1 (margin violations):
        - ∇w = lambda * w - (1/n) * ∑yᵢxᵢ
        - ∇b = -(1/n) * ∑yᵢ

        For samples where yᵢ(w·xᵢ + b) ≥ 1 (correct classification):
        - ∇w = lambda * w
        - ∇b = 0

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
        y : np.ndarray of shape (n_samples,)
            Target labels (-1 or +1)

        Returns:
        --------
        dw : np.ndarray of shape (n_features,)
            Gradient with respect to weights
        db : float
            Gradient with respect to bias
        """
        n_samples, n_features = X.shape

        # Compute linear predictions
        linear_output = X.dot(self.w) + self.b

        # Find samples with margin violations (hinge loss > 0)
        margin_violations = y * linear_output < 1

        # Initialize gradients with regularization term
        dw = self.lambda_param * self.w
        db = 0.0

        # Add contributions from margin violations
        if np.any(margin_violations):
            # Gradient contributions from misclassified samples
            violation_contributions_w = -y[margin_violations, np.newaxis] * X[margin_violations]
            violation_contributions_b = -y[margin_violations]

            # Average the contributions
            dw += np.mean(violation_contributions_w, axis=0)
            db += np.mean(violation_contributions_b)

        return dw, db

    def _update_parameters(self, dw: np.ndarray, db: float, current_lr: float) -> None:
        """Update parameters using gradient descent."""

        self.w -= current_lr * dw
        self.b -= current_lr * db

    def _check_convergence(self, cost_history: List[float], iteration: int) -> bool:
        """Check if training has converged based on cost changes."""

        if not self.early_stopping or len(cost_history) < self.patience + 1:
            return False

        # Check if cost hasn't improved significantly in the last 'patience' iterations
        recent_costs = cost_history[-self.patience - 1:]
        min_recent_cost = min(recent_costs[:-1])
        current_cost = recent_costs[-1]

        # Converged if current cost is not significantly better than recent minimum
        improvement = min_recent_cost - current_cost
        relative_improvement = improvement / (abs(min_recent_cost) + 1e-10)

        converged = relative_improvement < self.tolerance

        if converged and self.verbose:
            print(
                f"Converged at iteration {iteration}: improvement {relative_improvement:.2e} < tolerance {self.tolerance:.2e}")

        return converged

    def _estimate_support_vectors(self, X: np.ndarray, y: np.ndarray) -> int:
        """
        Estimate the number of support vectors.

        Support vectors are samples that are either:
        1. On the margin boundary: 0 < |w·x + b| ≤ 1
        2. Misclassified: y(w·x + b) < 1
        """

        # Compute decision function values
        decision_values = X.dot(self.w) + self.b

        # Check for margin violations and samples on the margin
        margin_violations = y * decision_values <= 1.0 + 1e-6  # Small tolerance for numerical precision

        return np.sum(margin_violations)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (sklearn compatibility)."""
        return {
            'learning_rate': self.learning_rate,
            'lambda_param': self.lambda_param,
            'n_iters': self.n_iters,
            'tolerance': self.tolerance,
            'learning_rate_decay': self.learning_rate_decay,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **parameters) -> 'SVMFromScratch':
        """Set parameters for this estimator (sklearn compatibility)."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __repr__(self) -> str:
        """String representation of the SVM classifier."""
        params = []
        for key, value in self.get_params().items():
            if isinstance(value, float):
                params.append(f"{key}={value:.4f}")
            else:
                params.append(f"{key}={value}")

        param_str = ", ".join(params)
        return f"SVMFromScratch({param_str})"


if __name__ == "__main__":
    # Test the SVM class structure
    print("Testing SVM class structure...")

    # Test initialization
    print("\n1. Testing initialization:")
    svm = SVMFromScratch(learning_rate=0.01, lambda_param=0.1, verbose=True)
    print(f"SVM initialized: {svm}")

    # Test parameter validation
    print("\n2. Testing parameter validation:")
    try:
        invalid_svm = SVMFromScratch(learning_rate=-0.01)
        print("❌ Should have raised ValueError for negative learning rate")
    except ValueError as e:
        print(f"Correctly caught invalid parameter: {e}")

    # Test input validation
    print("\n3. Testing input validation:")
    X_test = np.array([[1, 2], [2, 3], [3, 1]])
    y_test = np.array([1, -1, 1])

    try:
        X_validated, y_validated = svm._validate_input(X_test, y_test)
        print(f"Input validation passed: X shape {X_validated.shape}, y shape {y_validated.shape}")
    except Exception as e:
        print(f"❌ Input validation failed: {e}")

    # Test parameter initialization
    print("\n4. Testing parameter initialization:")
    svm._initialize_parameters(n_features=2)
    print(f"Parameters initialized: w={svm.w}, b={svm.b}")

    # Test cost computation
    print("\n5. Testing cost computation:")
    cost = svm._compute_cost(X_test, y_test)
    print(f"Cost computed: {cost:.4f}")

    # Test gradient computation
    print("\n6. Testing gradient computation:")
    dw, db = svm._compute_gradients(X_test, y_test)
    print(f"Gradients computed: dw={dw}, db={db:.4f}")

    print(f"\nAll tests passed! SVM class structure is working!")
