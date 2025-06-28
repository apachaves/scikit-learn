"""
Enhanced Histogram-based Gradient Boosting Demonstration
=========================================================

This example demonstrates the enhanced HistGradientBoosting estimators
with additional solvers, robust loss functions, and advanced features.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import (
    EnhancedHistGradientBoostingRegressor,
    EnhancedHistGradientBoostingClassifier,
)

print(__doc__)

# %%
# Regression Example with Outliers
# ---------------------------------

# Generate regression data with outliers
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Add outliers
outlier_indices = np.random.choice(len(y), size=50, replace=False)
y[outlier_indices] += np.random.normal(0, 10, size=50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standard HistGradientBoosting
standard_reg = HistGradientBoostingRegressor(max_iter=100, random_state=42)
standard_reg.fit(X_train, y_train)
y_pred_standard = standard_reg.predict(X_test)
mse_standard = mean_squared_error(y_test, y_pred_standard)
r2_standard = r2_score(y_test, y_pred_standard)

# Enhanced HistGradientBoosting with Huber loss (robust to outliers)
enhanced_reg = EnhancedHistGradientBoostingRegressor(
    loss='huber',
    solver='newton',
    l1_regularization=0.01,
    max_iter=100,
    random_state=42
)
enhanced_reg.fit(X_train, y_train)
y_pred_enhanced = enhanced_reg.predict(X_test)
mse_enhanced = mean_squared_error(y_test, y_pred_enhanced)
r2_enhanced = r2_score(y_test, y_pred_enhanced)

print("Regression Results (with outliers):")
print(f"Standard HGBR - MSE: {mse_standard:.4f}, R²: {r2_standard:.4f}")
print(f"Enhanced HGBR - MSE: {mse_enhanced:.4f}, R²: {r2_enhanced:.4f}")
print(f"Improvement: {((mse_standard - mse_enhanced) / mse_standard * 100):.1f}% MSE reduction")

# %%
# Classification Example with Imbalanced Data
# --------------------------------------------

# Generate imbalanced classification data
X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20, n_classes=2, 
    weights=[0.9, 0.1], random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print(f"\nClass distribution: {np.bincount(y_train_clf)}")

# Enhanced classifier with focal loss for imbalanced data
enhanced_clf = EnhancedHistGradientBoostingClassifier(
    loss='focal',
    solver='sgd',
    max_iter=100,
    random_state=42
)
enhanced_clf.fit(X_train_clf, y_train_clf)
y_pred_clf = enhanced_clf.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_clf)

print(f"Enhanced Classifier Accuracy: {accuracy:.4f}")

# %%
# Feature Importance Comparison
# -----------------------------

# Get different types of feature importance
importance_gain = enhanced_reg.get_feature_importance(method='gain')
importance_perm = enhanced_reg.get_feature_importance(
    method='permutation', X=X_test, y=y_test, n_repeats=3
)

print(f"\nFeature Importance Analysis:")
print(f"Gain-based importance shape: {importance_gain.shape}")
print(f"Top 3 features (gain): {np.argsort(importance_gain)[-3:]}")
print(f"Permutation importance available: {len(importance_perm['importances_mean'])}")

# %%
# Multi-output Regression Example
# -------------------------------

# Generate multi-output data
X_multi, y_single = make_regression(n_samples=500, n_features=10, random_state=42)
y_multi = np.column_stack([y_single, y_single * 0.5, y_single + 10])

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Multi-output enhanced regressor
multi_reg = EnhancedHistGradientBoostingRegressor(
    multi_output=True,
    max_iter=50,
    random_state=42
)
multi_reg.fit(X_train_multi, y_train_multi)
y_pred_multi = multi_reg.predict(X_test_multi)

print(f"\nMulti-output Regression:")
print(f"Prediction shape: {y_pred_multi.shape}")
for i in range(3):
    r2 = r2_score(y_test_multi[:, i], y_pred_multi[:, i])
    print(f"Output {i+1} R²: {r2:.4f}")

# %%
# Tree Statistics and Model Analysis
# ----------------------------------

stats = enhanced_reg.get_tree_statistics()
print(f"\nModel Statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")

# %%
# Visualization
# -------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Regression predictions comparison
axes[0, 0].scatter(y_test, y_pred_standard, alpha=0.6, label='Standard', color='blue')
axes[0, 0].scatter(y_test, y_pred_enhanced, alpha=0.6, label='Enhanced', color='red')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[0, 0].set_xlabel('True Values')
axes[0, 0].set_ylabel('Predictions')
axes[0, 0].set_title('Regression: Standard vs Enhanced')
axes[0, 0].legend()

# Plot 2: Feature importance
axes[0, 1].bar(range(len(importance_gain)), importance_gain)
axes[0, 1].set_xlabel('Feature Index')
axes[0, 1].set_ylabel('Importance')
axes[0, 1].set_title('Feature Importance (Gain-based)')

# Plot 3: Multi-output predictions
for i in range(3):
    axes[1, 0].scatter(y_test_multi[:, i], y_pred_multi[:, i], 
                      alpha=0.6, label=f'Output {i+1}')
axes[1, 0].plot([y_test_multi.min(), y_test_multi.max()], 
               [y_test_multi.min(), y_test_multi.max()], 'k--', lw=2)
axes[1, 0].set_xlabel('True Values')
axes[1, 0].set_ylabel('Predictions')
axes[1, 0].set_title('Multi-output Regression')
axes[1, 0].legend()

# Plot 4: Model complexity
tree_depths = [stats['avg_depth']] * stats['n_trees']
axes[1, 1].hist(tree_depths, bins=20, alpha=0.7)
axes[1, 1].set_xlabel('Tree Depth')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Tree Complexity (Avg Depth: {stats["avg_depth"]:.1f})')

plt.tight_layout()
plt.show()

print("\nEnhanced HistGradientBoosting demonstration completed!")
print("Key benefits demonstrated:")
print("- Improved robustness to outliers with Huber loss")
print("- Better handling of imbalanced data with focal loss")
print("- Multi-output regression capabilities")
print("- Advanced feature importance methods")
print("- Comprehensive model analysis tools")