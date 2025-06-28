# Enhanced HistGradientBoosting Usage Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Basic Usage](#basic-usage)
4. [Enhanced Features](#enhanced-features)
5. [Advanced Examples](#advanced-examples)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Introduction

The Enhanced HistGradientBoosting estimators extend scikit-learn's standard implementation with modern machine learning techniques, additional solvers, robust loss functions, and advanced interpretability features.

### Key Enhancements

- **Multiple Solvers**: Newton-Raphson, SGD, Coordinate Descent
- **Robust Loss Functions**: Huber, Focal, Custom losses
- **Advanced Regularization**: L1, Elastic Net, Dropout
- **Multi-output Support**: Handle multiple targets simultaneously
- **Enhanced Interpretability**: SHAP integration, advanced feature importance
- **Learning Rate Scheduling**: Cosine, exponential, adaptive schedules
- **Ensemble Diversity**: Bagging, random subspace methods
- **Robustness Features**: Outlier detection, noise injection

## Installation and Setup

### Prerequisites
```bash
pip install numpy scipy scikit-learn
pip install shap  # Optional, for SHAP-based interpretability
```

### Basic Import
```python
from enhanced_hist_gradient_boosting import (
    EnhancedHistGradientBoostingRegressor,
    EnhancedHistGradientBoostingClassifier,
    EnhancedBoostingConfig
)
```

## Basic Usage

### Regression Example
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train enhanced regressor
regressor = EnhancedHistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    random_state=42
)

regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

### Classification Example
```python
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train enhanced classifier
classifier = EnhancedHistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    random_state=42
)

classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

## Enhanced Features

### 1. Alternative Solvers

#### Newton-Raphson Solver
```python
# Enhanced second-order optimization
regressor = EnhancedHistGradientBoostingRegressor(
    solver='newton',
    max_iter=100,
    random_state=42
)
```

#### Stochastic Gradient Descent
```python
# Mini-batch training for large datasets
regressor = EnhancedHistGradientBoostingRegressor(
    solver='sgd',
    enhanced_config=EnhancedBoostingConfig(
        solver_params={'subsample_size': 0.8, 'momentum': 0.9}
    ),
    max_iter=100,
    random_state=42
)
```

### 2. Robust Loss Functions

#### Huber Loss (Robust to Outliers)
```python
# Robust regression with Huber loss
regressor = EnhancedHistGradientBoostingRegressor(
    loss='huber',
    max_iter=100,
    random_state=42
)

# Add outliers to demonstrate robustness
y_train_outliers = y_train.copy()
outlier_indices = np.random.choice(len(y_train), size=50, replace=False)
y_train_outliers[outlier_indices] += np.random.normal(0, 10, size=50)

regressor.fit(X_train, y_train_outliers)
```

#### Focal Loss (Imbalanced Classification)
```python
# Handle imbalanced classification with focal loss
classifier = EnhancedHistGradientBoostingClassifier(
    loss='focal',
    max_iter=100,
    random_state=42
)
```

#### Custom Loss Functions
```python
def asymmetric_loss(y_true, y_pred, sample_weight=None, alpha=0.7):
    """Custom asymmetric loss function."""
    residual = y_true - y_pred
    loss = np.where(residual >= 0, 
                   alpha * residual**2, 
                   (1-alpha) * residual**2)
    
    gradient = np.where(residual >= 0,
                       -2 * alpha * residual,
                       -2 * (1-alpha) * residual)
    
    hessian = np.where(residual >= 0,
                      2 * alpha * np.ones_like(residual),
                      2 * (1-alpha) * np.ones_like(residual))
    
    if sample_weight is not None:
        loss *= sample_weight
        gradient *= sample_weight
        hessian *= sample_weight
        
    return loss.mean(), gradient, hessian

# Use custom loss
regressor = EnhancedHistGradientBoostingRegressor(
    loss=asymmetric_loss,
    max_iter=100,
    random_state=42
)
```

### 3. Enhanced Regularization

```python
# L1 and L2 regularization
regressor = EnhancedHistGradientBoostingRegressor(
    l1_regularization=0.01,      # L1 (Lasso) regularization
    l2_regularization=0.01,      # L2 (Ridge) regularization
    enhanced_config=EnhancedBoostingConfig(
        elastic_net_ratio=0.5,   # Elastic net mixing
        dropout_rate=0.1,        # Tree dropout
        feature_dropout_rate=0.05 # Feature dropout
    ),
    max_iter=100,
    random_state=42
)
```

### 4. Learning Rate Scheduling

```python
# Cosine annealing schedule
regressor = EnhancedHistGradientBoostingRegressor(
    learning_rate_schedule='cosine',
    enhanced_config=EnhancedBoostingConfig(
        initial_learning_rate=0.1,
        final_learning_rate=0.01
    ),
    max_iter=100,
    random_state=42
)

# Exponential decay
regressor = EnhancedHistGradientBoostingRegressor(
    learning_rate_schedule='exponential',
    enhanced_config=EnhancedBoostingConfig(
        initial_learning_rate=0.1,
        schedule_params={'decay_rate': 0.95}
    ),
    max_iter=100,
    random_state=42
)
```

### 5. Ensemble Diversity Methods

```python
# Bagging and feature subsampling
regressor = EnhancedHistGradientBoostingRegressor(
    bagging=True,
    enhanced_config=EnhancedBoostingConfig(
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_bagging_fraction=0.8,
        random_subspace=True,
        subspace_size=0.7
    ),
    max_iter=100,
    random_state=42
)
```

### 6. Multi-output Regression

```python
# Generate multi-output data
X, y_single = make_regression(n_samples=1000, n_features=20, random_state=42)
y_multi = np.column_stack([y_single, y_single * 0.5, y_single + 10])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi, test_size=0.2, random_state=42
)

# Multi-output regressor
regressor = EnhancedHistGradientBoostingRegressor(
    multi_output=True,
    enhanced_config=EnhancedBoostingConfig(
        output_correlation='joint'  # Model output correlations
    ),
    max_iter=100,
    random_state=42
)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)  # Shape: (n_samples, 3)

print(f"Multi-output predictions shape: {y_pred.shape}")
```

### 7. Advanced Feature Importance

```python
# Train model
regressor.fit(X_train, y_train)

# Gain-based importance (default)
importance_gain = regressor.get_feature_importance(method='gain')

# Permutation importance
importance_perm = regressor.get_feature_importance(
    method='permutation', 
    X=X_test, 
    y=y_test, 
    n_repeats=5
)

# SHAP importance (requires shap package)
try:
    importance_shap = regressor.get_feature_importance(
        method='shap', 
        X=X_test[:100]  # Use subset for speed
    )
    print("SHAP importance calculated successfully")
except ImportError:
    print("SHAP not available, install with: pip install shap")

print(f"Gain importance: {importance_gain}")
print(f"Permutation importance mean: {importance_perm['importances_mean']}")
```

### 8. Model Interpretability

```python
# Get comprehensive tree statistics
stats = regressor.get_tree_statistics()
print("Tree Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Local explanations (requires shap)
try:
    local_explanations = regressor.explain_local(X_test[:5])
    print(f"Local explanations shape: {local_explanations.shape}")
except ImportError:
    print("SHAP required for local explanations")
```

### 9. Robustness Features

```python
# Outlier detection and noise injection
regressor = EnhancedHistGradientBoostingRegressor(
    outlier_detection=True,
    enhanced_config=EnhancedBoostingConfig(
        outlier_threshold=0.1,
        noise_injection=True,
        noise_level=0.01
    ),
    max_iter=100,
    random_state=42
)
```

## Advanced Examples

### Example 1: Comprehensive Enhanced Regressor

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate complex dataset with outliers
X, y = make_regression(n_samples=2000, n_features=50, noise=0.1, random_state=42)

# Add outliers
outlier_indices = np.random.choice(len(y), size=100, replace=False)
y[outlier_indices] += np.random.normal(0, 20, size=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create comprehensive enhanced configuration
config = EnhancedBoostingConfig(
    solver='newton',
    l1_regularization=0.01,
    elastic_net_ratio=0.5,
    learning_rate_schedule='cosine',
    initial_learning_rate=0.1,
    final_learning_rate=0.01,
    bagging=True,
    bagging_fraction=0.8,
    feature_bagging_fraction=0.9,
    outlier_detection=True,
    outlier_threshold=0.05,
    noise_injection=True,
    noise_level=0.005,
    feature_importance_method='shap'
)

# Create enhanced regressor
regressor = EnhancedHistGradientBoostingRegressor(
    loss='huber',  # Robust to outliers
    solver='newton',
    l1_regularization=0.01,
    learning_rate_schedule='cosine',
    bagging=True,
    outlier_detection=True,
    enhanced_config=config,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=1
)

# Fit model
regressor.fit(X_train, y_train)

# Evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Enhanced Regressor Results:")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Get detailed analysis
stats = regressor.get_tree_statistics()
print(f"Trees built: {stats['n_trees']}")
print(f"Average depth: {stats['avg_depth']:.2f}")

# Feature importance
importance = regressor.get_feature_importance(method='gain')
top_features = np.argsort(importance)[-10:]
print(f"Top 10 features: {top_features}")
```

### Example 2: Imbalanced Classification with Focal Loss

```python
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.datasets import make_imbalance  # pip install imbalanced-learn

# Create imbalanced dataset
X, y = make_classification(
    n_samples=2000, n_features=20, n_classes=2, 
    n_informative=15, n_redundant=5, random_state=42
)

# Make it more imbalanced
X_imb, y_imb = make_imbalance(
    X, y, sampling_strategy={0: 1600, 1: 400}, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
)

print(f"Class distribution: {np.bincount(y_train)}")

# Enhanced classifier with focal loss
classifier = EnhancedHistGradientBoostingClassifier(
    loss='focal',  # Better for imbalanced data
    solver='sgd',
    learning_rate_schedule='cosine',
    enhanced_config=EnhancedBoostingConfig(
        solver_params={'subsample_size': 0.8},
        initial_learning_rate=0.1,
        final_learning_rate=0.01,
        bagging=True,
        bagging_fraction=0.7
    ),
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
    verbose=1
)

classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)[:, 1]

print("Classification Results:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

### Example 3: Multi-output Regression with Correlated Targets

```python
# Generate correlated multi-output data
np.random.seed(42)
X = np.random.randn(1000, 20)

# Create correlated targets
y1 = X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.1
y2 = 0.5 * y1 + X[:, 3] - X[:, 4] + np.random.randn(1000) * 0.1
y3 = -0.3 * y1 + 0.7 * y2 + X[:, 5] + np.random.randn(1000) * 0.1

y_multi = np.column_stack([y1, y2, y3])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi, test_size=0.2, random_state=42
)

# Multi-output regressor that models correlations
regressor = EnhancedHistGradientBoostingRegressor(
    multi_output=True,
    enhanced_config=EnhancedBoostingConfig(
        output_correlation='joint',  # Model output correlations
        solver='newton',
        l1_regularization=0.005,
        l2_regularization=0.005
    ),
    max_iter=150,
    learning_rate=0.1,
    random_state=42
)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluate each output
for i in range(3):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    print(f"Output {i+1} R²: {r2:.4f}")

# Overall performance
overall_r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
print(f"Overall R²: {overall_r2:.4f}")
```

## Performance Optimization

### 1. Memory Optimization

```python
# For large datasets, use memory-efficient settings
regressor = EnhancedHistGradientBoostingRegressor(
    enhanced_config=EnhancedBoostingConfig(
        memory_efficient=True,
        low_memory_mode=True
    ),
    max_bins=128,  # Reduce memory usage
    max_iter=100,
    random_state=42
)
```

### 2. Speed Optimization

```python
# For faster training
regressor = EnhancedHistGradientBoostingRegressor(
    solver='sgd',  # Faster for large datasets
    enhanced_config=EnhancedBoostingConfig(
        solver_params={'subsample_size': 0.5}  # Use smaller subsamples
    ),
    max_leaf_nodes=15,  # Smaller trees
    max_iter=50,  # Fewer iterations
    early_stopping=True,
    n_iter_no_change=10,
    random_state=42
)
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'solver': ['standard', 'newton', 'sgd'],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_iter': [50, 100, 200],
    'l1_regularization': [0.0, 0.01, 0.1],
    'l2_regularization': [0.0, 0.01, 0.1]
}

# Grid search
regressor = EnhancedHistGradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    regressor, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
```python
# Problem: Out of memory errors
# Solution: Use memory-efficient settings
regressor = EnhancedHistGradientBoostingRegressor(
    enhanced_config=EnhancedBoostingConfig(
        memory_efficient=True,
        low_memory_mode=True
    ),
    max_bins=64,  # Reduce bins
    max_leaf_nodes=15  # Smaller trees
)
```

#### 2. Slow Training
```python
# Problem: Training takes too long
# Solution: Use faster solver and early stopping
regressor = EnhancedHistGradientBoostingRegressor(
    solver='sgd',
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1
)
```

#### 3. Overfitting
```python
# Problem: Model overfits to training data
# Solution: Increase regularization
regressor = EnhancedHistGradientBoostingRegressor(
    l1_regularization=0.1,
    l2_regularization=0.1,
    enhanced_config=EnhancedBoostingConfig(
        dropout_rate=0.2,
        feature_dropout_rate=0.1
    ),
    max_leaf_nodes=10,
    min_samples_leaf=50
)
```

#### 4. Poor Performance on Imbalanced Data
```python
# Problem: Poor performance on minority class
# Solution: Use focal loss and class weights
classifier = EnhancedHistGradientBoostingClassifier(
    loss='focal',
    class_weight='balanced'
)
```

#### 5. SHAP Import Errors
```python
# Problem: SHAP not available
# Solution: Install SHAP or use alternative importance methods
try:
    importance = regressor.get_feature_importance(method='shap', X=X_test)
except ImportError:
    print("SHAP not available, using permutation importance")
    importance = regressor.get_feature_importance(
        method='permutation', X=X_test, y=y_test
    )
```

## API Reference

### EnhancedHistGradientBoostingRegressor

#### Parameters
- `loss` : str, BaseLoss, or callable
  - Loss function to use
  - Options: 'squared_error', 'absolute_error', 'huber', 'gamma', 'poisson', 'quantile'
  
- `solver` : str, default='standard'
  - Optimization solver
  - Options: 'standard', 'newton', 'sgd', 'coordinate'
  
- `l1_regularization` : float, default=0.0
  - L1 regularization parameter
  
- `learning_rate_schedule` : str, default='constant'
  - Learning rate scheduling strategy
  - Options: 'constant', 'cosine', 'exponential', 'step', 'adaptive'
  
- `bagging` : bool, default=False
  - Whether to use bagging for ensemble diversity
  
- `outlier_detection` : bool, default=False
  - Whether to enable automatic outlier detection
  
- `multi_output` : bool, default=False
  - Whether to enable multi-output regression
  
- `enhanced_config` : EnhancedBoostingConfig, optional
  - Advanced configuration object

#### Methods
- `fit(X, y, sample_weight=None)`
  - Fit the enhanced gradient boosting model
  
- `predict(X)`
  - Predict using the enhanced model
  
- `get_feature_importance(method='gain', **kwargs)`
  - Get feature importance using specified method
  - Methods: 'gain', 'permutation', 'shap'
  
- `explain_local(X, method='shap', **kwargs)`
  - Explain individual predictions
  
- `get_tree_statistics()`
  - Get comprehensive tree statistics

### EnhancedHistGradientBoostingClassifier

Similar to the regressor but adapted for classification tasks.

#### Additional Parameters
- `class_weight` : dict, 'balanced', or None
  - Weights associated with classes

### EnhancedBoostingConfig

Configuration dataclass for enhanced features.

#### Key Parameters
- `solver` : str
- `l1_regularization` : float
- `learning_rate_schedule` : str
- `bagging` : bool
- `outlier_detection` : bool
- `multi_output` : bool
- `feature_importance_method` : str

## Conclusion

The Enhanced HistGradientBoosting estimators provide a comprehensive set of modern machine learning techniques while maintaining compatibility with the scikit-learn ecosystem. The modular design allows users to gradually adopt enhanced features based on their specific needs and requirements.

For more examples and advanced usage patterns, refer to the test suite and example notebooks provided with the implementation.