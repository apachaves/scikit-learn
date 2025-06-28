"""
Enhanced Histogram-based Gradient Boosting Estimators

This module provides enhanced versions of HistGradientBoosting estimators with
additional solvers, loss functions, and advanced features.
"""

from .enhanced_gradient_boosting import (
    EnhancedHistGradientBoostingRegressor,
    EnhancedHistGradientBoostingClassifier,
    EnhancedBoostingConfig,
)

__all__ = [
    "EnhancedHistGradientBoostingRegressor",
    "EnhancedHistGradientBoostingClassifier", 
    "EnhancedBoostingConfig",
]