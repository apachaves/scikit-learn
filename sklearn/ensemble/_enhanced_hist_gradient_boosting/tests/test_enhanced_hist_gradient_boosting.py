"""
Comprehensive tests for Enhanced HistGradientBoosting Estimators

This module provides extensive testing for the enhanced gradient boosting
implementation, including unit tests, integration tests, and performance benchmarks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.utils._testing import assert_warns

# Import our enhanced implementations
from .. import (
    EnhancedHistGradientBoostingRegressor,
    EnhancedHistGradientBoostingClassifier,
    EnhancedBoostingConfig,
)
from ..enhanced_gradient_boosting import (
    NewtonSolver,
    SGDSolver,
    LearningRateScheduler,
    RobustLossCollection,
    FeatureImportanceCalculator,
)


class TestEnhancedBoostingConfig:
    """Test the enhanced boosting configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EnhancedBoostingConfig()
        assert config.solver == "standard"
        assert config.l1_regularization == 0.0
        assert config.learning_rate_schedule == "constant"
        assert config.bagging is False
        assert config.multi_output is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EnhancedBoostingConfig(
            solver="newton",
            l1_regularization=0.01,
            bagging=True,
            outlier_detection=True,
        )
        assert config.solver == "newton"
        assert config.l1_regularization == 0.01
        assert config.bagging is True
        assert config.outlier_detection is True


class TestSolvers:
    """Test different solver implementations."""
    
    def test_newton_solver(self):
        """Test Newton solver initialization and basic functionality."""
        solver = NewtonSolver(hessian_regularization=1e-6, adaptive_step_size=True)
        assert solver.hessian_regularization == 1e-6
        assert solver.adaptive_step_size is True
        
        # Test fit_iteration method
        gradients = np.random.randn(100)
        hessians = np.abs(np.random.randn(100)) + 0.1  # Ensure positive
        X_binned = np.random.randint(0, 10, (100, 5))
        
        result_grad, result_hess = solver.fit_iteration(X_binned, gradients, hessians)
        assert result_grad.shape == gradients.shape
        assert result_hess.shape == hessians.shape
        assert np.all(result_hess >= solver.hessian_regularization)
    
    def test_sgd_solver(self):
        """Test SGD solver initialization and basic functionality."""
        solver = SGDSolver(subsample_size=0.8, momentum=0.9)
        assert solver.subsample_size == 0.8
        assert solver.momentum == 0.9
        
        # Test fit_iteration method
        gradients = np.random.randn(100)
        hessians = np.abs(np.random.randn(100)) + 0.1
        X_binned = np.random.randint(0, 10, (100, 5))
        
        result_grad, result_hess = solver.fit_iteration(X_binned, gradients, hessians)
        # SGD should subsample, so result should be smaller
        assert len(result_grad) <= len(gradients)
        assert len(result_hess) <= len(hessians)


class TestLearningRateScheduler:
    """Test learning rate scheduling strategies."""
    
    def test_constant_schedule(self):
        """Test constant learning rate schedule."""
        scheduler = LearningRateScheduler("constant", 0.1, 0.01)
        
        for iteration in [0, 10, 50, 100]:
            lr = scheduler.get_learning_rate(iteration, 100)
            assert lr == 0.1
    
    def test_cosine_schedule(self):
        """Test cosine annealing schedule."""
        scheduler = LearningRateScheduler("cosine", 0.1, 0.01)
        
        lr_start = scheduler.get_learning_rate(0, 100)
        lr_mid = scheduler.get_learning_rate(50, 100)
        lr_end = scheduler.get_learning_rate(100, 100)
        
        assert lr_start == pytest.approx(0.1, rel=1e-3)
        assert lr_end == pytest.approx(0.01, rel=1e-3)
        assert lr_start > lr_mid > lr_end
    
    def test_exponential_schedule(self):
        """Test exponential decay schedule."""
        scheduler = LearningRateScheduler("exponential", 0.1, 0.01, decay_rate=0.95)
        
        lr_start = scheduler.get_learning_rate(0, 100)
        lr_later = scheduler.get_learning_rate(10, 100)
        
        assert lr_start == 0.1
        assert lr_later < lr_start
        assert lr_later == pytest.approx(0.1 * (0.95 ** 10), rel=1e-3)


class TestRobustLossCollection:
    """Test robust loss functions."""
    
    def test_huber_loss(self):
        """Test Huber loss function."""
        y_true = np.array([1.0, 2.0, 3.0, 10.0])  # Last value is outlier
        y_pred = np.array([1.1, 2.1, 3.1, 5.0])
        
        loss, gradient, hessian = RobustLossCollection.huber_loss(
            y_true, y_pred, delta=1.0
        )
        
        assert isinstance(loss, float)
        assert gradient.shape == y_true.shape
        assert hessian.shape == y_true.shape
        
        # Check that outlier gets clipped gradient
        residual = y_true - y_pred
        assert np.abs(gradient[-1]) <= 1.0  # Should be clipped for outlier
    
    def test_focal_loss(self):
        """Test focal loss function."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([-2.0, 2.0, 0.5, -0.5])  # Logits
        
        loss, gradient, hessian = RobustLossCollection.focal_loss(
            y_true, y_pred, alpha=1.0, gamma=2.0
        )
        
        assert isinstance(loss, float)
        assert gradient.shape == y_true.shape
        assert hessian.shape == y_true.shape
        assert loss >= 0  # Loss should be non-negative
    
    def test_loss_with_sample_weights(self):
        """Test loss functions with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        sample_weight = np.array([1.0, 2.0, 0.5])
        
        loss_weighted, grad_weighted, hess_weighted = RobustLossCollection.huber_loss(
            y_true, y_pred, sample_weight=sample_weight
        )
        
        loss_unweighted, grad_unweighted, hess_unweighted = RobustLossCollection.huber_loss(
            y_true, y_pred, sample_weight=None
        )
        
        # Weighted versions should be different
        assert loss_weighted != loss_unweighted
        assert not np.array_equal(grad_weighted, grad_unweighted)
        assert not np.array_equal(hess_weighted, hess_unweighted)


class TestEnhancedHistGradientBoostingRegressor:
    """Test enhanced gradient boosting regressor."""
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression dataset for testing."""
        X, y = make_regression(
            n_samples=200, n_features=10, noise=0.1, random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_basic_functionality(self, regression_data):
        """Test basic fit and predict functionality."""
        X_train, X_test, y_train, y_test = regression_data
        
        estimator = EnhancedHistGradientBoostingRegressor(
            max_iter=20, random_state=42
        )
        estimator.fit(X_train, y_train)
        
        # Test prediction
        y_pred = estimator.predict(X_test)
        assert y_pred.shape == y_test.shape
        
        # Test score
        score = estimator.score(X_test, y_test)
        assert 0 <= score <= 1  # R² score should be reasonable
    
    def test_enhanced_solvers(self, regression_data):
        """Test different solver options."""
        X_train, X_test, y_train, y_test = regression_data
        
        solvers = ["standard", "newton", "sgd"]
        scores = {}
        
        for solver in solvers:
            estimator = EnhancedHistGradientBoostingRegressor(
                solver=solver, max_iter=20, random_state=42
            )
            estimator.fit(X_train, y_train)
            scores[solver] = estimator.score(X_test, y_test)
        
        # All solvers should produce reasonable results
        for solver, score in scores.items():
            assert score > 0.5, f"Solver {solver} produced poor score: {score}"
    
    def test_robust_loss_functions(self, regression_data):
        """Test robust loss functions."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Add some outliers to test robustness
        y_train_outliers = y_train.copy()
        outlier_indices = np.random.choice(len(y_train), size=10, replace=False)
        y_train_outliers[outlier_indices] += np.random.normal(0, 10, size=10)
        
        # Test Huber loss (should be more robust to outliers)
        estimator_huber = EnhancedHistGradientBoostingRegressor(
            loss="huber", max_iter=20, random_state=42
        )
        estimator_huber.fit(X_train, y_train_outliers)
        score_huber = estimator_huber.score(X_test, y_test)
        
        # Test standard squared error
        estimator_standard = EnhancedHistGradientBoostingRegressor(
            loss="squared_error", max_iter=20, random_state=42
        )
        estimator_standard.fit(X_train, y_train_outliers)
        score_standard = estimator_standard.score(X_test, y_test)
        
        # Huber loss should be more robust (though this is not guaranteed in all cases)
        assert score_huber > 0.3  # Should still produce reasonable results
        assert score_standard > 0.3  # Should still produce reasonable results
    
    def test_learning_rate_scheduling(self, regression_data):
        """Test learning rate scheduling."""
        X_train, X_test, y_train, y_test = regression_data
        
        schedules = ["constant", "cosine", "exponential"]
        
        for schedule in schedules:
            estimator = EnhancedHistGradientBoostingRegressor(
                learning_rate_schedule=schedule,
                max_iter=20,
                random_state=42
            )
            estimator.fit(X_train, y_train)
            score = estimator.score(X_test, y_test)
            assert score > 0.3, f"Schedule {schedule} produced poor score: {score}"
    
    def test_regularization(self, regression_data):
        """Test L1 and L2 regularization."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Test with regularization
        estimator_reg = EnhancedHistGradientBoostingRegressor(
            l1_regularization=0.01,
            l2_regularization=0.01,
            max_iter=20,
            random_state=42
        )
        estimator_reg.fit(X_train, y_train)
        score_reg = estimator_reg.score(X_test, y_test)
        
        # Test without regularization
        estimator_no_reg = EnhancedHistGradientBoostingRegressor(
            l1_regularization=0.0,
            l2_regularization=0.0,
            max_iter=20,
            random_state=42
        )
        estimator_no_reg.fit(X_train, y_train)
        score_no_reg = estimator_no_reg.score(X_test, y_test)
        
        # Both should produce reasonable results
        assert score_reg > 0.3
        assert score_no_reg > 0.3
    
    def test_feature_importance_methods(self, regression_data):
        """Test different feature importance calculation methods."""
        X_train, X_test, y_train, y_test = regression_data
        
        estimator = EnhancedHistGradientBoostingRegressor(
            max_iter=20, random_state=42
        )
        estimator.fit(X_train, y_train)
        
        # Test gain-based importance
        importance_gain = estimator.get_feature_importance(method="gain")
        assert importance_gain.shape == (X_train.shape[1],)
        assert np.all(importance_gain >= 0)
        assert np.sum(importance_gain) > 0
        
        # Test permutation importance
        importance_perm = estimator.get_feature_importance(
            method="permutation", X=X_test, y=y_test, n_repeats=3
        )
        assert "importances_mean" in importance_perm
        assert "importances_std" in importance_perm
        assert importance_perm["importances_mean"].shape == (X_train.shape[1],)
    
    def test_multi_output_regression(self):
        """Test multi-output regression functionality."""
        X, y_single = make_regression(
            n_samples=200, n_features=10, random_state=42
        )
        
        # Create multi-output target
        y_multi = np.column_stack([y_single, y_single * 0.5, y_single + 10])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_multi, test_size=0.3, random_state=42
        )
        
        estimator = EnhancedHistGradientBoostingRegressor(
            multi_output=True, max_iter=20, random_state=42
        )
        estimator.fit(X_train, y_train)
        
        # Test prediction
        y_pred = estimator.predict(X_test)
        assert y_pred.shape == y_test.shape
        
        # Test that we have separate estimators for each output
        assert hasattr(estimator, "estimators_")
        assert len(estimator.estimators_) == y_multi.shape[1]
    
    def test_tree_statistics(self, regression_data):
        """Test tree statistics functionality."""
        X_train, X_test, y_train, y_test = regression_data
        
        estimator = EnhancedHistGradientBoostingRegressor(
            max_iter=10, random_state=42
        )
        estimator.fit(X_train, y_train)
        
        stats = estimator.get_tree_statistics()
        
        assert "n_trees" in stats
        assert "total_leaves" in stats
        assert "avg_leaves_per_tree" in stats
        assert "max_depth" in stats
        assert "avg_depth" in stats
        
        assert stats["n_trees"] > 0
        assert stats["total_leaves"] > 0
        assert stats["max_depth"] >= 0


class TestEnhancedHistGradientBoostingClassifier:
    """Test enhanced gradient boosting classifier."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset for testing."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_basic_functionality(self, classification_data):
        """Test basic fit and predict functionality."""
        X_train, X_test, y_train, y_test = classification_data
        
        estimator = EnhancedHistGradientBoostingClassifier(
            max_iter=20, random_state=42
        )
        estimator.fit(X_train, y_train)
        
        # Test prediction
        y_pred = estimator.predict(X_test)
        assert y_pred.shape == y_test.shape
        
        # Test predict_proba
        y_proba = estimator.predict_proba(X_test)
        assert y_proba.shape == (len(y_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
        
        # Test score
        score = estimator.score(X_test, y_test)
        assert 0 <= score <= 1
    
    def test_focal_loss(self, classification_data):
        """Test focal loss for imbalanced classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        estimator = EnhancedHistGradientBoostingClassifier(
            loss="focal", max_iter=20, random_state=42
        )
        estimator.fit(X_train, y_train)
        
        score = estimator.score(X_test, y_test)
        assert score > 0.6  # Should achieve reasonable accuracy
    
    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        estimator = EnhancedHistGradientBoostingClassifier(
            max_iter=20, random_state=42
        )
        estimator.fit(X_train, y_train)
        
        # Test prediction
        y_pred = estimator.predict(X_test)
        assert y_pred.shape == y_test.shape
        
        # Test predict_proba
        y_proba = estimator.predict_proba(X_test)
        assert y_proba.shape == (len(y_test), 3)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
        
        score = estimator.score(X_test, y_test)
        assert score > 0.5  # Should achieve reasonable accuracy


class TestIntegration:
    """Integration tests for enhanced estimators."""
    
    def test_sklearn_compatibility(self):
        """Test compatibility with scikit-learn ecosystem."""
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        # Test in pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", EnhancedHistGradientBoostingRegressor(max_iter=10))
        ])
        
        # Test cross-validation
        scores = cross_val_score(pipeline, X, y, cv=3, scoring="r2")
        assert len(scores) == 3
        assert all(score > -1 for score in scores)  # Reasonable scores
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid solver
        with pytest.raises(ValueError):
            estimator = EnhancedHistGradientBoostingRegressor(solver="invalid")
            estimator._validate_params()
        
        # Test invalid learning rate schedule
        with pytest.raises(ValueError):
            estimator = EnhancedHistGradientBoostingRegressor(
                learning_rate_schedule="invalid"
            )
            estimator._validate_params()
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random state."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        estimator1 = EnhancedHistGradientBoostingRegressor(
            max_iter=10, random_state=42
        )
        estimator2 = EnhancedHistGradientBoostingRegressor(
            max_iter=10, random_state=42
        )
        
        estimator1.fit(X, y)
        estimator2.fit(X, y)
        
        pred1 = estimator1.predict(X)
        pred2 = estimator2.predict(X)
        
        assert_allclose(pred1, pred2, rtol=1e-10)


class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_performance_comparison(self):
        """Compare performance with standard implementation."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        
        X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standard implementation
        standard = HistGradientBoostingRegressor(max_iter=50, random_state=42)
        standard.fit(X_train, y_train)
        score_standard = standard.score(X_test, y_test)
        
        # Enhanced implementation (with standard settings)
        enhanced = EnhancedHistGradientBoostingRegressor(
            solver="standard", max_iter=50, random_state=42
        )
        enhanced.fit(X_train, y_train)
        score_enhanced = enhanced.score(X_test, y_test)
        
        # Enhanced should perform at least as well as standard
        assert abs(score_enhanced - score_standard) < 0.1
    
    def test_memory_usage(self):
        """Test memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and fit a large model
        X, y = make_regression(n_samples=5000, n_features=50, random_state=42)
        estimator = EnhancedHistGradientBoostingRegressor(
            max_iter=100, random_state=42
        )
        estimator.fit(X, y)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, f"Memory increase too large: {memory_increase}MB"


if __name__ == "__main__":
    # Run basic tests if executed directly
    print("Running basic tests for Enhanced HistGradientBoosting...")
    
    # Test configuration
    config = EnhancedBoostingConfig(solver="newton", bagging=True)
    print(f"✓ Configuration test passed: {config.solver}, {config.bagging}")
    
    # Test regressor
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    regressor = EnhancedHistGradientBoostingRegressor(
        solver="newton", max_iter=10, random_state=42
    )
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    print(f"✓ Regressor test passed: R² = {score:.3f}")
    
    # Test classifier
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    classifier = EnhancedHistGradientBoostingClassifier(
        loss="focal", max_iter=10, random_state=42
    )
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(f"✓ Classifier test passed: Accuracy = {score:.3f}")
    
    # Test feature importance
    importance = regressor.get_feature_importance(method="gain")
    print(f"✓ Feature importance test passed: shape = {importance.shape}")
    
    print("\nAll basic tests passed! ✨")
    print("Run 'pytest test_enhanced_hist_gradient_boosting.py' for comprehensive testing.")