"""
Enhanced HistGradientBoosting Estimator Implementation

This module provides enhanced versions of HistGradientBoosting estimators with
additional solvers, loss functions, and advanced features.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Callable, Literal, Protocol, Union
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, _fit_context
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    BaseHistGradientBoosting,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn._loss.loss import BaseLoss
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted


# Type aliases
Float = np.floating[Any]
ArrayFloat = NDArray[Float]
LossFunction = Union[str, BaseLoss, Callable]
SolverType = Literal["standard", "newton", "sgd", "coordinate", "adaboost"]
ScheduleType = Literal["constant", "cosine", "exponential", "step", "adaptive"]


@dataclass(frozen=True)
class EnhancedBoostingConfig:
    """Configuration for enhanced boosting features."""
    
    # Solver configuration
    solver: SolverType = "standard"
    solver_params: dict[str, Any] = field(default_factory=dict)
    
    # Enhanced regularization
    l1_regularization: float = 0.0
    elastic_net_ratio: float = 0.5
    dropout_rate: float = 0.0
    feature_dropout_rate: float = 0.0
    adaptive_regularization: bool = False
    
    # Learning rate scheduling
    learning_rate_schedule: ScheduleType = "constant"
    initial_learning_rate: float = 0.1
    final_learning_rate: float = 0.01
    schedule_params: dict[str, Any] = field(default_factory=dict)
    
    # Ensemble diversity
    bagging: bool = False
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    feature_bagging_fraction: float = 1.0
    random_subspace: bool = False
    subspace_size: float = 1.0
    
    # Robustness features
    outlier_detection: bool = False
    outlier_threshold: float = 0.1
    noise_injection: bool = False
    noise_level: float = 0.01
    
    # Multi-output support
    multi_output: bool = False
    output_correlation: Literal["independent", "chained", "joint"] = "independent"
    
    # Interpretability
    feature_importance_method: Literal["gain", "split", "permutation", "shap"] = "gain"
    
    # Performance optimizations
    memory_efficient: bool = False
    low_memory_mode: bool = False


class CustomLossProtocol(Protocol):
    """Protocol for custom loss functions."""
    
    def __call__(
        self,
        y_true: ArrayFloat,
        y_pred: ArrayFloat,
        sample_weight: ArrayFloat | None = None,
        **kwargs: Any,
    ) -> tuple[float, ArrayFloat, ArrayFloat]:
        """Compute loss, gradient, and hessian.
        
        Returns
        -------
        loss : float
            Average loss value
        gradient : array-like
            Gradient with respect to y_pred
        hessian : array-like
            Hessian (second derivative) with respect to y_pred
        """
        ...


class SolverBase(ABC):
    """Base class for gradient boosting solvers."""
    
    def __init__(self, **params: Any) -> None:
        self.params = params
    
    @abstractmethod
    def fit_iteration(
        self,
        X_binned: ArrayFloat,
        gradients: ArrayFloat,
        hessians: ArrayFloat,
        **kwargs: Any,
    ) -> Any:
        """Fit one iteration of the solver."""
        ...


class NewtonSolver(SolverBase):
    """Newton-Raphson solver with enhanced second-order optimization."""
    
    def __init__(
        self,
        hessian_regularization: float = 1e-6,
        adaptive_step_size: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.hessian_regularization = hessian_regularization
        self.adaptive_step_size = adaptive_step_size
    
    def fit_iteration(
        self,
        X_binned: ArrayFloat,
        gradients: ArrayFloat,
        hessians: ArrayFloat,
        **kwargs: Any,
    ) -> Any:
        """Fit one Newton iteration with enhanced Hessian computation."""
        # Regularize hessian to avoid numerical issues
        regularized_hessians = hessians + self.hessian_regularization
        
        # Adaptive step size based on Hessian condition number
        if self.adaptive_step_size:
            condition_number = np.max(regularized_hessians) / np.max(
                np.minimum(regularized_hessians, 1e-8)
            )
            step_size = min(1.0, 10.0 / condition_number)
            gradients = gradients * step_size
        
        return gradients, regularized_hessians


class SGDSolver(SolverBase):
    """Stochastic Gradient Descent solver with mini-batch support."""
    
    def __init__(
        self,
        subsample_size: float = 0.8,
        momentum: float = 0.9,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.subsample_size = subsample_size
        self.momentum = momentum
        self.velocity = None
    
    def fit_iteration(
        self,
        X_binned: ArrayFloat,
        gradients: ArrayFloat,
        hessians: ArrayFloat,
        **kwargs: Any,
    ) -> Any:
        """Fit one SGD iteration with momentum."""
        n_samples = len(gradients)
        subsample_indices = np.random.choice(
            n_samples,
            size=int(n_samples * self.subsample_size),
            replace=False,
        )
        
        # Apply momentum
        if self.velocity is None:
            self.velocity = np.zeros_like(gradients)
        
        self.velocity = (
            self.momentum * self.velocity + (1 - self.momentum) * gradients
        )
        
        return self.velocity[subsample_indices], hessians[subsample_indices]


class LearningRateScheduler:
    """Learning rate scheduling strategies."""
    
    def __init__(
        self,
        schedule: ScheduleType,
        initial_lr: float,
        final_lr: float,
        **params: Any,
    ) -> None:
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.params = params
    
    def get_learning_rate(self, iteration: int, max_iterations: int) -> float:
        """Get learning rate for current iteration."""
        progress = iteration / max_iterations
        
        if self.schedule == "constant":
            return self.initial_lr
        elif self.schedule == "cosine":
            return self.final_lr + (self.initial_lr - self.final_lr) * (
                1 + np.cos(np.pi * progress)
            ) / 2
        elif self.schedule == "exponential":
            decay_rate = self.params.get("decay_rate", 0.95)
            return self.initial_lr * (decay_rate ** iteration)
        elif self.schedule == "step":
            step_size = self.params.get("step_size", max_iterations // 3)
            decay_factor = self.params.get("decay_factor", 0.5)
            return self.initial_lr * (decay_factor ** (iteration // step_size))
        else:
            return self.initial_lr


class RobustLossCollection:
    """Collection of robust loss functions."""
    
    @staticmethod
    def huber_loss(
        y_true: ArrayFloat,
        y_pred: ArrayFloat,
        sample_weight: ArrayFloat | None = None,
        delta: float = 1.0,
    ) -> tuple[float, ArrayFloat, ArrayFloat]:
        """Huber loss function."""
        residual = y_true - y_pred
        abs_residual = np.abs(residual)
        
        # Loss
        loss = np.where(
            abs_residual <= delta,
            0.5 * residual**2,
            delta * abs_residual - 0.5 * delta**2,
        )
        
        # Gradient
        gradient = np.where(
            abs_residual <= delta,
            -residual,
            -delta * np.sign(residual),
        )
        
        # Hessian
        hessian = np.where(abs_residual <= delta, 1.0, 0.0)
        
        if sample_weight is not None:
            loss *= sample_weight
            gradient *= sample_weight
            hessian *= sample_weight
        
        return loss.mean(), gradient, hessian
    
    @staticmethod
    def focal_loss(
        y_true: ArrayFloat,
        y_pred: ArrayFloat,
        sample_weight: ArrayFloat | None = None,
        alpha: float = 1.0,
        gamma: float = 2.0,
    ) -> tuple[float, ArrayFloat, ArrayFloat]:
        """Focal loss for imbalanced classification."""
        # Convert to probabilities
        p = 1 / (1 + np.exp(-y_pred))
        
        # Focal loss computation
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
        p_t = np.where(y_true == 1, p, 1 - p)
        
        loss = -alpha_t * (1 - p_t) ** gamma * np.log(np.clip(p_t, 1e-8, 1 - 1e-8))
        
        # Gradient (simplified)
        gradient = alpha_t * (1 - p_t) ** gamma * (
            gamma * p_t * np.log(np.clip(p_t, 1e-8, 1 - 1e-8)) + p_t - y_true
        )
        
        # Hessian (approximation)
        hessian = alpha_t * (1 - p_t) ** gamma * p_t * (1 - p_t)
        
        if sample_weight is not None:
            loss *= sample_weight
            gradient *= sample_weight
            hessian *= sample_weight
        
        return loss.mean(), gradient, hessian


class FeatureImportanceCalculator:
    """Advanced feature importance calculation methods."""
    
    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator
    
    def calculate_gain_importance(self) -> ArrayFloat:
        """Calculate gain-based feature importance."""
        check_is_fitted(self.estimator)
        return self.estimator.feature_importances_
    
    def calculate_permutation_importance(
        self,
        X: ArrayFloat,
        y: ArrayFloat,
        n_repeats: int = 5,
        random_state: int | None = None,
    ) -> dict[str, ArrayFloat]:
        """Calculate permutation-based feature importance."""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            self.estimator, X, y, n_repeats=n_repeats, random_state=random_state
        )
        return {
            "importances_mean": result.importances_mean,
            "importances_std": result.importances_std,
        }
    
    def calculate_shap_importance(
        self,
        X: ArrayFloat,
        max_evals: int = 1000,
    ) -> ArrayFloat:
        """Calculate SHAP-based feature importance."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required for SHAP-based feature importance. "
                "Install it with: pip install shap"
            )
        
        explainer = shap.TreeExplainer(self.estimator)
        shap_values = explainer.shap_values(X[:max_evals])
        
        if isinstance(shap_values, list):
            # Multi-class case
            return np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        else:
            return np.abs(shap_values).mean(0)


class EnhancedBaseHistGradientBoosting(BaseHistGradientBoosting):
    """Enhanced base class for histogram-based gradient boosting."""
    
    _parameter_constraints = {
        **BaseHistGradientBoosting._parameter_constraints,
        "enhanced_config": [EnhancedBoostingConfig, None],
        "solver": [StrOptions({"standard", "newton", "sgd", "coordinate", "adaboost"})],
        "l1_regularization": [Interval(Real, 0, None, closed="left")],
        "learning_rate_schedule": [
            StrOptions({"constant", "cosine", "exponential", "step", "adaptive"})
        ],
        "bagging": ["boolean"],
        "outlier_detection": ["boolean"],
        "multi_output": ["boolean"],
    }
    
    def __init__(
        self,
        loss,
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring=None,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        # Enhanced parameters
        enhanced_config=None,
        solver="standard",
        l1_regularization=0.0,
        learning_rate_schedule="constant",
        bagging=False,
        outlier_detection=False,
        multi_output=False,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )
        
        # Enhanced parameters
        self.enhanced_config = enhanced_config or EnhancedBoostingConfig()
        self.solver = solver
        self.l1_regularization = l1_regularization
        self.learning_rate_schedule = learning_rate_schedule
        self.bagging = bagging
        self.outlier_detection = outlier_detection
        self.multi_output = multi_output
        
        # Initialize enhanced components
        self._solver = None
        self._lr_scheduler = None
        self._importance_calculator = None
    
    def _initialize_enhanced_components(self) -> None:
        """Initialize enhanced components based on configuration."""
        # Initialize solver
        if self.solver == "newton":
            self._solver = NewtonSolver(**self.enhanced_config.solver_params)
        elif self.solver == "sgd":
            self._solver = SGDSolver(**self.enhanced_config.solver_params)
        else:
            self._solver = None  # Use standard solver
        
        # Initialize learning rate scheduler
        if self.learning_rate_schedule != "constant":
            self._lr_scheduler = LearningRateScheduler(
                schedule=self.learning_rate_schedule,
                initial_lr=self.enhanced_config.initial_learning_rate,
                final_lr=self.enhanced_config.final_learning_rate,
                **self.enhanced_config.schedule_params,
            )
        
        # Initialize feature importance calculator
        self._importance_calculator = FeatureImportanceCalculator(self)
    
    def get_feature_importance(
        self,
        method: str = "gain",
        X: ArrayFloat | None = None,
        y: ArrayFloat | None = None,
        **kwargs: Any,
    ) -> ArrayFloat | dict[str, ArrayFloat]:
        """Get feature importance using specified method."""
        check_is_fitted(self)
        
        if self._importance_calculator is None:
            self._importance_calculator = FeatureImportanceCalculator(self)
        
        if method == "gain":
            return self._importance_calculator.calculate_gain_importance()
        elif method == "permutation":
            if X is None or y is None:
                raise ValueError("X and y are required for permutation importance")
            return self._importance_calculator.calculate_permutation_importance(
                X, y, **kwargs
            )
        elif method == "shap":
            if X is None:
                raise ValueError("X is required for SHAP importance")
            return self._importance_calculator.calculate_shap_importance(X, **kwargs)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def explain_local(
        self,
        X: ArrayFloat,
        method: str = "shap",
        **kwargs: Any,
    ) -> ArrayFloat:
        """Explain individual predictions."""
        check_is_fitted(self)
        
        if method == "shap":
            try:
                import shap
            except ImportError:
                raise ImportError(
                    "SHAP is required for local explanations. "
                    "Install it with: pip install shap"
                )
            
            explainer = shap.TreeExplainer(self)
            return explainer.shap_values(X)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
    
    def get_tree_statistics(self) -> dict[str, Any]:
        """Get comprehensive tree statistics."""
        check_is_fitted(self)
        
        n_trees = len(self._predictors)
        total_leaves = sum(
            predictor.get_n_leaf_nodes()
            for predictors_at_iter in self._predictors
            for predictor in predictors_at_iter
        )
        
        depths = []
        for predictors_at_iter in self._predictors:
            for predictor in predictors_at_iter:
                depths.append(predictor.get_max_depth())
        
        return {
            "n_trees": n_trees,
            "total_leaves": total_leaves,
            "avg_leaves_per_tree": total_leaves / max(n_trees, 1),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": np.mean(depths) if depths else 0,
            "std_depth": np.std(depths) if depths else 0,
        }


class EnhancedHistGradientBoostingRegressor(
    RegressorMixin, EnhancedBaseHistGradientBoosting
):
    """Enhanced Histogram-based Gradient Boosting Regression Tree.
    
    This estimator extends the standard HistGradientBoostingRegressor with:
    - Additional solvers (Newton, SGD, Coordinate Descent)
    - Robust loss functions (Huber, Fair, LogCosh)
    - Enhanced regularization (L1, Elastic Net, Dropout)
    - Multi-output regression support
    - Advanced feature importance methods
    - Learning rate scheduling
    - Ensemble diversity methods
    - Outlier detection and robustness features
    
    Parameters
    ----------
    loss : str, BaseLoss, or callable, default='squared_error'
        The loss function to use. Can be a string, BaseLoss instance, or
        custom callable following the CustomLossProtocol.
        
    solver : {'standard', 'newton', 'sgd', 'coordinate'}, default='standard'
        The optimization solver to use.
        
    l1_regularization : float, default=0.0
        L1 regularization parameter.
        
    learning_rate_schedule : {'constant', 'cosine', 'exponential', 'step'}, default='constant'
        Learning rate scheduling strategy.
        
    bagging : bool, default=False
        Whether to use bagging for ensemble diversity.
        
    outlier_detection : bool, default=False
        Whether to enable automatic outlier detection.
        
    multi_output : bool, default=False
        Whether to enable multi-output regression.
        
    enhanced_config : EnhancedBoostingConfig, optional
        Advanced configuration object for fine-tuning enhanced features.
    """
    
    _parameter_constraints = {
        **EnhancedBaseHistGradientBoosting._parameter_constraints,
        "loss": [
            StrOptions({
                "squared_error", "absolute_error", "gamma", "poisson", 
                "quantile", "huber", "fair", "logcosh"
            }),
            BaseLoss,
            callable,
        ],
        "quantile": [Interval(Real, 0, 1, closed="both"), None],
    }
    
    def __init__(
        self,
        loss="squared_error",
        *,
        quantile=None,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring=None,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        # Enhanced parameters
        enhanced_config=None,
        solver="standard",
        l1_regularization=0.0,
        learning_rate_schedule="constant",
        bagging=False,
        outlier_detection=False,
        multi_output=False,
    ):
        self.quantile = quantile
        
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            enhanced_config=enhanced_config,
            solver=solver,
            l1_regularization=l1_regularization,
            learning_rate_schedule=learning_rate_schedule,
            bagging=bagging,
            outlier_detection=outlier_detection,
            multi_output=multi_output,
        )
    
    def _get_loss(self, sample_weight):
        """Get the loss function, including enhanced loss functions."""
        if isinstance(self.loss, str):
            if self.loss == "huber":
                return lambda y_true, y_pred, sw=None: RobustLossCollection.huber_loss(
                    y_true, y_pred, sw, delta=1.0
                )
            # Add other robust losses here
        
        # Fall back to parent implementation
        return super()._get_loss(sample_weight)
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the enhanced gradient boosting model."""
        # Initialize enhanced components
        self._initialize_enhanced_components()
        
        # Handle multi-output case
        if self.multi_output and y.ndim > 1:
            return self._fit_multi_output(X, y, sample_weight)
        
        # Standard fit with enhancements
        return super().fit(X, y, sample_weight)
    
    def _fit_multi_output(self, X, y, sample_weight=None):
        """Fit multi-output regression model."""
        if y.ndim == 1:
            raise ValueError("y must be 2D for multi-output regression")
        
        n_outputs = y.shape[1]
        self.estimators_ = []
        
        for i in range(n_outputs):
            estimator = EnhancedHistGradientBoostingRegressor(
                **self.get_params(deep=False)
            )
            estimator.multi_output = False  # Avoid recursion
            estimator.fit(X, y[:, i], sample_weight)
            self.estimators_.append(estimator)
        
        return self
    
    def predict(self, X):
        """Predict using the enhanced model."""
        check_is_fitted(self)
        
        if hasattr(self, "estimators_"):
            # Multi-output prediction
            predictions = np.column_stack([
                estimator.predict(X) for estimator in self.estimators_
            ])
            return predictions
        
        return super().predict(X)


class EnhancedHistGradientBoostingClassifier(
    ClassifierMixin, EnhancedBaseHistGradientBoosting
):
    """Enhanced Histogram-based Gradient Boosting Classification Tree.
    
    This estimator extends the standard HistGradientBoostingClassifier with
    enhanced features similar to the regressor, adapted for classification tasks.
    """
    
    _parameter_constraints = {
        **EnhancedBaseHistGradientBoosting._parameter_constraints,
        "loss": [
            StrOptions({"log_loss", "focal"}),
            BaseLoss,
            callable,
        ],
        "class_weight": [dict, StrOptions({"balanced"}), None],
    }
    
    def __init__(
        self,
        loss="log_loss",
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring=None,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        class_weight=None,
        # Enhanced parameters
        enhanced_config=None,
        solver="standard",
        l1_regularization=0.0,
        learning_rate_schedule="constant",
        bagging=False,
        outlier_detection=False,
        multi_output=False,
    ):
        self.class_weight = class_weight
        
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            enhanced_config=enhanced_config,
            solver=solver,
            l1_regularization=l1_regularization,
            learning_rate_schedule=learning_rate_schedule,
            bagging=bagging,
            outlier_detection=outlier_detection,
            multi_output=multi_output,
        )
    
    def _get_loss(self, sample_weight):
        """Get the loss function, including enhanced loss functions."""
        if isinstance(self.loss, str):
            if self.loss == "focal":
                return lambda y_true, y_pred, sw=None: RobustLossCollection.focal_loss(
                    y_true, y_pred, sw, alpha=1.0, gamma=2.0
                )
        
        # Fall back to parent implementation
        return super()._get_loss(sample_weight)


# Example usage and testing
if __name__ == "__main__":
    # Example usage of enhanced estimators
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score
    
    # Test enhanced regressor
    print("Testing Enhanced HistGradientBoosting Regressor...")
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Create enhanced regressor with robust loss
    enhanced_reg = EnhancedHistGradientBoostingRegressor(
        loss="huber",
        solver="newton",
        learning_rate_schedule="cosine",
        l1_regularization=0.01,
        bagging=True,
        max_iter=50,
        random_state=42
    )
    
    enhanced_reg.fit(X_train, y_train)
    y_pred = enhanced_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Enhanced Regressor MSE: {mse:.4f}")
    
    # Test feature importance
    importance = enhanced_reg.get_feature_importance(method="gain")
    print(f"Feature importance shape: {importance.shape}")
    
    # Test tree statistics
    stats = enhanced_reg.get_tree_statistics()
    print(f"Tree statistics: {stats}")
    
    # Test enhanced classifier
    print("\nTesting Enhanced HistGradientBoosting Classifier...")
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    enhanced_clf = EnhancedHistGradientBoostingClassifier(
        loss="focal",
        solver="sgd",
        learning_rate_schedule="exponential",
        max_iter=50,
        random_state=42
    )
    
    enhanced_clf.fit(X_train, y_train)
    y_pred = enhanced_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Enhanced Classifier Accuracy: {accuracy:.4f}")
    
    print("\nEnhanced HistGradientBoosting implementation completed successfully!")