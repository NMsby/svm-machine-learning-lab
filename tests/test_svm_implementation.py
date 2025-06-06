"""
Unit tests for SVM from Scratch implementation
"""

import numpy as np
import pytest
import sys
import os

# Add src to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from svm_from_scratch import SVMFromScratch


class TestSVMFromScratch:
    """Test suite for SVMFromScratch class"""

    def setup_method(self):
        """Set up test data for each test method"""
        np.random.seed(42)

        # Simple linearly separable data
        n_samples = 50
        X1 = np.random.normal([2, 2], 0.5, (n_samples // 2, 2))
        X2 = np.random.normal([-2, -2], 0.5, (n_samples // 2, 2))
        self.X_simple = np.vstack([X1, X2])
        self.y_simple = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])

        # Shuffle
        shuffle_idx = np.random.permutation(len(self.X_simple))
        self.X_simple = self.X_simple[shuffle_idx]
        self.y_simple = self.y_simple[shuffle_idx]

    def test_initialization(self):
        """Test SVM initialization with various parameters"""

        # Test default initialization
        svm = SVMFromScratch()
        assert svm.learning_rate == 0.01
        assert svm.lambda_param == 0.01
        assert svm.n_iters == 1000

        # Test custom parameters
        svm_custom = SVMFromScratch(
            learning_rate=0.001,
            lambda_param=0.1,
            n_iters=500
        )
        assert svm_custom.learning_rate == 0.001
        assert svm_custom.lambda_param == 0.1
        assert svm_custom.n_iters == 500

    def test_parameter_validation(self):
        """Test parameter validation in initialization"""

        # Test invalid learning rate
        with pytest.raises(ValueError):
            SVMFromScratch(learning_rate=-0.01)

        # Test invalid lambda parameter
        with pytest.raises(ValueError):
            SVMFromScratch(lambda_param=0)

        # Test invalid number of iterations
        with pytest.raises(ValueError):
            SVMFromScratch(n_iters=-100)

    def test_input_validation(self):
        """Test input validation in fit method"""
        svm = SVMFromScratch()

        # Test wrong dimensions
        X_wrong = np.array([1, 2, 3])  # 1D instead of 2D
        y_wrong = np.array([-1, 1])

        with pytest.raises(ValueError):
            svm.fit(X_wrong, y_wrong)

        # Test mismatched dimensions
        X_mismatch = np.array([[1, 2], [3, 4]])
        y_mismatch = np.array([-1])  # Wrong length

        with pytest.raises(ValueError):
            svm.fit(X_mismatch, y_mismatch)

    def test_basic_training(self):
        """Test basic training functionality"""
        svm = SVMFromScratch(n_iters=100, verbose=False)

        # Should not raise any exceptions
        svm.fit(self.X_simple, self.y_simple)

        # Check that model parameters are set
        assert svm.w is not None
        assert svm.b is not None
        assert len(svm.costs) > 0
        assert len(svm.w) == self.X_simple.shape[1]

    def test_predictions(self):
        """Test prediction functionality"""
        svm = SVMFromScratch(n_iters=200, verbose=False)
        svm.fit(self.X_simple, self.y_simple)

        # Test predictions
        predictions = svm.predict(self.X_simple)
        assert len(predictions) == len(self.y_simple)
        assert np.all(np.isin(predictions, [-1, 1]))

        # Test decision function
        decisions = svm.decision_function(self.X_simple)
        assert len(decisions) == len(self.y_simple)

        # Predictions should match sign of decision function
        assert np.array_equal(predictions, np.where(decisions >= 0, 1, -1))

    def test_probability_predictions(self):
        """Test probability prediction functionality"""
        svm = SVMFromScratch(n_iters=200, verbose=False)
        svm.fit(self.X_simple, self.y_simple)

        probas = svm.predict_proba(self.X_simple)

        # Check shape and properties
        assert probas.shape == (len(self.X_simple), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probas >= 0) and np.all(probas <= 1)  # Valid probabilities

    def test_scoring(self):
        """Test accuracy scoring"""
        svm = SVMFromScratch(n_iters=300, verbose=False)
        svm.fit(self.X_simple, self.y_simple)

        accuracy = svm.score(self.X_simple, self.y_simple)
        assert 0 <= accuracy <= 1

        # For linearly separable data, should achieve reasonable accuracy
        assert accuracy > 0.7  # Should be much higher, but being conservative

    def test_cost_computation(self):
        """Test cost computation"""
        svm = SVMFromScratch(n_iters=10, verbose=False)
        svm.fit(self.X_simple, self.y_simple)

        # Check that costs are computed
        assert len(svm.costs) == 10
        assert all(cost >= 0 for cost in svm.costs)  # Costs should be non-negative

    def test_convergence(self):
        """Test convergence detection"""
        svm = SVMFromScratch(
            n_iters=1000,
            early_stopping=True,
            tolerance=1e-4,
            verbose=False
        )
        svm.fit(self.X_simple, self.y_simple)

        # Should converge before max iterations for simple data
        assert len(svm.costs) < 1000 or svm.converged

    def test_get_params(self):
        """Test parameter getter"""
        params = {
            'learning_rate': 0.005,
            'lambda_param': 0.05,
            'n_iters': 300,
            'tolerance': 1e-5,
            'early_stopping': False,
            'verbose': True
        }

        svm = SVMFromScratch(**params)
        retrieved_params = svm.get_params()

        for key, value in params.items():
            assert retrieved_params[key] == value

    def test_single_sample_prediction(self):
        """Test prediction on single sample"""
        svm = SVMFromScratch(n_iters=100, verbose=False)
        svm.fit(self.X_simple, self.y_simple)

        # Test single sample (1D array)
        single_sample = self.X_simple[0]
        prediction = svm.predict(single_sample)
        decision = svm.decision_function(single_sample)

        assert len(prediction) == 1
        assert len(decision) == 1
        assert prediction[0] in [-1, 1]

    def test_different_label_formats(self):
        """Test handling of different label formats"""
        svm = SVMFromScratch(n_iters=50, verbose=False)

        # Test with 0/1 labels (should be converted automatically)
        y_binary = np.where(self.y_simple == -1, 0, 1)

        with pytest.warns(UserWarning):  # Should warn about label conversion
            svm.fit(self.X_simple, y_binary)

        # Should still work correctly
        predictions = svm.predict(self.X_simple)
        assert np.all(np.isin(predictions, [-1, 1]))


def run_tests():
    """Run all tests manually"""
    test_instance = TestSVMFromScratch()

    print("Running SVM implementation tests...")
    print("=" * 50)

    test_methods = [
        method for method in dir(test_instance)
        if method.startswith('test_')
    ]

    passed = 0
    failed = 0

    for test_method in test_methods:
        try:
            test_instance.setup_method()
            getattr(test_instance, test_method)()
            print(f"✅ {test_method}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method}: FAILED - {str(e)}")
            failed += 1

    print("-" * 50)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed!")
    else:
        print("Some tests failed. Please check implementation.")

    return failed == 0


if __name__ == "__main__":
    run_tests()
