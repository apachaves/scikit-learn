# Example of modernized typing for scikit-learn
# This shows how to enhance type hints and use modern Python features

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, Self, TypeVar
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

# Type aliases for better readability
Float = np.floating[Any]
Int = np.integer[Any]
ArrayFloat = NDArray[Float]
ArrayInt = NDArray[Int]

# Generic type variables
EstimatorType = TypeVar("EstimatorType", bound="BaseEstimator")
TargetType = TypeVar("TargetType", bound=ArrayLike)

# Protocol for estimators that can predict
class Predictor(Protocol):
    """Protocol for estimators that implement predict method."""
    
    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict using the fitted model."""
        ...

# Modern dataclass for configuration
@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for machine learning models.
    
    This replaces traditional parameter dictionaries with a type-safe,
    immutable configuration object.
    """
    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6
    random_state: int | None = None
    verbose: bool = False
    
    # Class variable for validation
    _valid_solvers: ClassVar[frozenset[str]] = frozenset({"lbfgs", "sgd", "adam"})
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")

# Enhanced base class with modern typing
class ModernBaseEstimator(ABC):
    """Modern base estimator with enhanced type hints."""
    
    def __init__(
        self,
        *,
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or ModelConfig()
        self._is_fitted = False
        
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the estimator to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        ...
    
    @abstractmethod
    def predict(self, X: ArrayLike) -> ArrayFloat:
        """Make predictions on new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        ...
    
    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        """Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from sklearn.metrics import r2_score
        
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)
    
    def save_model(self, path: str | Path) -> None:
        """Save the fitted model to disk.
        
        Uses joblib for safe serialization instead of pickle.
        
        Parameters
        ----------
        path : str or Path
            Path where to save the model.
        """
        import joblib
        
        if not self._is_fitted:
            warnings.warn("Saving unfitted model", UserWarning, stacklevel=2)
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use joblib for safer serialization
        joblib.dump(self, path, compress=3)
    
    @classmethod
    def load_model(cls, path: str | Path) -> Self:
        """Load a fitted model from disk.
        
        Parameters
        ----------
        path : str or Path
            Path to the saved model.
            
        Returns
        -------
        model : object
            The loaded model instance.
        """
        import joblib
        
        return joblib.load(path)

# Example of modern parameter validation
def validate_array_input(
    X: ArrayLike,
    *,
    dtype: type[Float] | Literal["numeric"] = "numeric",
    ensure_2d: bool = True,
    allow_nd: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
) -> ArrayFloat:
    """Validate and convert input arrays with modern type hints.
    
    Parameters
    ----------
    X : array-like
        Input array to validate.
    dtype : data-type or "numeric", default="numeric"
        Data type to enforce.
    ensure_2d : bool, default=True
        Whether to ensure the array is 2D.
    allow_nd : bool, default=False
        Whether to allow n-dimensional arrays.
    ensure_min_samples : int, default=1
        Minimum number of samples required.
    ensure_min_features : int, default=1
        Minimum number of features required.
        
    Returns
    -------
    X_validated : ndarray
        Validated and converted array.
        
    Raises
    ------
    ValueError
        If validation fails.
    """
    from sklearn.utils.validation import check_array
    
    return check_array(
        X,
        dtype=dtype,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
    )

# Example of using structural pattern matching (Python 3.10+)
def handle_solver_config(solver: str, config: dict[str, Any]) -> dict[str, Any]:
    """Configure solver parameters using pattern matching.
    
    This demonstrates Python 3.10+ structural pattern matching.
    """
    match solver:
        case "lbfgs":
            return {
                "max_iter": config.get("max_iter", 1000),
                "tolerance": config.get("tolerance", 1e-6),
                "memory": config.get("memory", 10),
            }
        case "sgd":
            return {
                "learning_rate": config.get("learning_rate", 0.01),
                "momentum": config.get("momentum", 0.9),
                "nesterov": config.get("nesterov", True),
            }
        case "adam":
            return {
                "learning_rate": config.get("learning_rate", 0.001),
                "beta1": config.get("beta1", 0.9),
                "beta2": config.get("beta2", 0.999),
                "epsilon": config.get("epsilon", 1e-8),
            }
        case _:
            raise ValueError(f"Unknown solver: {solver}")

# Example of enhanced error handling with context
class ModelError(Exception):
    """Base exception for model-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.context = context or {}

class FittingError(ModelError):
    """Exception raised during model fitting."""
    pass

class PredictionError(ModelError):
    """Exception raised during prediction."""
    pass

# Example of using context managers for resource management
from contextlib import contextmanager
from typing import Generator

@contextmanager
def temporary_config(
    estimator: ModernBaseEstimator,
    **temp_config: Any,
) -> Generator[ModernBaseEstimator, None, None]:
    """Temporarily modify estimator configuration.
    
    This is useful for testing different configurations without
    permanently modifying the estimator.
    """
    original_config = estimator.config
    try:
        # Create new config with temporary values
        new_config_dict = {
            **original_config.__dict__,
            **temp_config,
        }
        estimator.config = ModelConfig(**new_config_dict)
        yield estimator
    finally:
        estimator.config = original_config