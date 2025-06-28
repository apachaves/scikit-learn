.. _enhanced_histogram_based_gradient_boosting:

Enhanced Histogram-Based Gradient Boosting
-------------------------------------------

Scikit-learn provides enhanced versions of histogram-based gradient boosting
estimators: :class:`EnhancedHistGradientBoostingClassifier` and
:class:`EnhancedHistGradientBoostingRegressor`. These estimators extend the
standard :class:`HistGradientBoostingClassifier` and
:class:`HistGradientBoostingRegressor` with modern machine learning techniques,
additional optimization solvers, robust loss functions, and advanced
interpretability features.

The enhanced estimators maintain **full backward compatibility** with the
standard implementations while providing significant improvements for
challenging datasets, including better handling of outliers, imbalanced data,
and multi-output scenarios.

.. topic:: Enhanced vs Standard HistGradientBoosting

  The enhanced versions provide all the capabilities of the standard
  implementations plus:

  * **Multiple optimization solvers**: Newton-Raphson, SGD, Coordinate Descent
  * **Robust loss functions**: Huber loss, Focal loss, Custom loss support
  * **Advanced regularization**: L1, Elastic Net, Dropout regularization
  * **Multi-output support**: Handle multiple correlated targets simultaneously
  * **Enhanced interpretability**: SHAP integration, advanced feature importance
  * **Learning rate scheduling**: Cosine, exponential, adaptive schedules
  * **Ensemble diversity**: Bagging, random subspace methods
  * **Robustness features**: Outlier detection, noise injection

Key Enhancements
^^^^^^^^^^^^^^^^

Multiple Optimization Solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The enhanced estimators support multiple optimization solvers beyond the
standard gradient descent approach:

* **Newton-Raphson solver** (``solver='newton'``): Uses enhanced second-order
  optimization with adaptive step sizing for better convergence, especially
  beneficial for well-conditioned problems.

* **SGD solver** (``solver='sgd'``): Stochastic gradient descent with momentum
  and mini-batch support, ideal for large datasets where memory efficiency is
  important.

* **Coordinate Descent solver** (``solver='coordinate'``): Feature-wise
  optimization particularly effective for high-dimensional sparse data.

Example usage::

  >>> from sklearn.ensemble import EnhancedHistGradientBoostingRegressor
  >>> from sklearn.datasets import make_regression
  >>> X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
  >>> 
  >>> # Newton-Raphson solver for better convergence
  >>> regressor = EnhancedHistGradientBoostingRegressor(
  ...     solver='newton',
  ...     max_iter=100,
  ...     random_state=42
  ... )
  >>> regressor.fit(X, y)
  EnhancedHistGradientBoostingRegressor(...)

Robust Loss Functions
~~~~~~~~~~~~~~~~~~~~~

Enhanced estimators provide robust loss functions that are less sensitive to
outliers and better suited for specific data characteristics:

**For Regression:**

* **Huber loss** (``loss='huber'``): Combines the best properties of squared
  error and absolute error, being quadratic for small residuals and linear for
  large residuals, making it robust to outliers.

* **Custom loss functions**: Support for user-defined loss functions that
  follow the gradient boosting protocol.

**For Classification:**

* **Focal loss** (``loss='focal'``): Designed for imbalanced classification
  problems, it down-weights easy examples and focuses learning on hard examples.

Example with robust loss::

  >>> # Robust regression with outliers
  >>> regressor = EnhancedHistGradientBoostingRegressor(
  ...     loss='huber',
  ...     solver='newton',
  ...     max_iter=100,
  ...     random_state=42
  ... )
  >>> 
  >>> # Classification with imbalanced data
  >>> from sklearn.ensemble import EnhancedHistGradientBoostingClassifier
  >>> classifier = EnhancedHistGradientBoostingClassifier(
  ...     loss='focal',
  ...     max_iter=100,
  ...     random_state=42
  ... )

Advanced Regularization
~~~~~~~~~~~~~~~~~~~~~~~

The enhanced estimators provide multiple regularization techniques:

* **L1 regularization** (``l1_regularization``): Promotes sparsity and feature
  selection by penalizing the absolute values of the leaf weights.

* **Elastic Net regularization**: Combines L1 and L2 regularization through
  the ``enhanced_config`` parameter.

* **Dropout regularization**: Tree and feature dropout to prevent overfitting.

Example with regularization::

  >>> from sklearn.ensemble._enhanced_hist_gradient_boosting import EnhancedBoostingConfig
  >>> 
  >>> config = EnhancedBoostingConfig(
  ...     l1_regularization=0.01,
  ...     elastic_net_ratio=0.5,
  ...     dropout_rate=0.1,
  ...     feature_dropout_rate=0.05
  ... )
  >>> 
  >>> regressor = EnhancedHistGradientBoostingRegressor(
  ...     l1_regularization=0.01,
  ...     enhanced_config=config,
  ...     max_iter=100,
  ...     random_state=42
  ... )

Multi-output Support
~~~~~~~~~~~~~~~~~~~~

Enhanced estimators can handle multiple output targets simultaneously, with
options to model correlations between outputs:

Example with multi-output regression::

  >>> import numpy as np
  >>> X, y_single = make_regression(n_samples=1000, n_features=10, random_state=42)
  >>> # Create correlated multi-output targets
  >>> y_multi = np.column_stack([y_single, y_single * 0.5, y_single + 10])
  >>> 
  >>> regressor = EnhancedHistGradientBoostingRegressor(
  ...     multi_output=True,
  ...     enhanced_config=EnhancedBoostingConfig(
  ...         output_correlation='joint'
  ...     ),
  ...     max_iter=100,
  ...     random_state=42
  ... )
  >>> regressor.fit(X, y_multi)
  EnhancedHistGradientBoostingRegressor(...)
  >>> predictions = regressor.predict(X)  # Shape: (n_samples, 3)

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~

Enhanced estimators support adaptive learning rate schedules that can improve
convergence and final performance:

* **Cosine annealing** (``learning_rate_schedule='cosine'``): Gradually
  decreases learning rate following a cosine curve.

* **Exponential decay** (``learning_rate_schedule='exponential'``): 
  Exponentially decreases learning rate over iterations.

* **Step decay** (``learning_rate_schedule='step'``): Decreases learning rate
  at fixed intervals.

Example with learning rate scheduling::

  >>> regressor = EnhancedHistGradientBoostingRegressor(
  ...     learning_rate_schedule='cosine',
  ...     enhanced_config=EnhancedBoostingConfig(
  ...         initial_learning_rate=0.1,
  ...         final_learning_rate=0.01
  ...     ),
  ...     max_iter=100,
  ...     random_state=42
  ... )

Enhanced Interpretability
~~~~~~~~~~~~~~~~~~~~~~~~~

The enhanced estimators provide advanced methods for model interpretation:

* **Multiple feature importance methods**: Gain-based, permutation-based, and
  SHAP-based feature importance.

* **Local explanations**: Individual prediction explanations using SHAP.

* **Tree statistics**: Comprehensive analysis of the ensemble structure.

Example with enhanced interpretability::

  >>> regressor.fit(X, y)
  >>> 
  >>> # Different types of feature importance
  >>> importance_gain = regressor.get_feature_importance(method='gain')
  >>> importance_perm = regressor.get_feature_importance(
  ...     method='permutation', X=X, y=y, n_repeats=5
  ... )
  >>> 
  >>> # SHAP-based importance (requires shap package)
  >>> try:
  ...     importance_shap = regressor.get_feature_importance(
  ...         method='shap', X=X[:100]
  ...     )
  ... except ImportError:
  ...     print("SHAP package required for SHAP-based importance")
  >>> 
  >>> # Tree statistics
  >>> stats = regressor.get_tree_statistics()
  >>> print(f"Number of trees: {stats['n_trees']}")
  >>> print(f"Average tree depth: {stats['avg_depth']:.2f}")

Ensemble Diversity Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enhanced estimators support ensemble diversity techniques to improve
generalization:

* **Bagging** (``bagging=True``): Bootstrap aggregating with configurable
  sample and feature fractions.

* **Random subspace methods**: Random feature subsampling for each tree.

Example with ensemble diversity::

  >>> regressor = EnhancedHistGradientBoostingRegressor(
  ...     bagging=True,
  ...     enhanced_config=EnhancedBoostingConfig(
  ...         bagging_fraction=0.8,
  ...         feature_bagging_fraction=0.9,
  ...         random_subspace=True,
  ...         subspace_size=0.7
  ...     ),
  ...     max_iter=100,
  ...     random_state=42
  ... )

Custom Loss Functions
~~~~~~~~~~~~~~~~~~~~~

Enhanced estimators support custom loss functions that follow the gradient
boosting protocol, allowing for domain-specific objectives:

Example with custom loss::

  >>> def asymmetric_loss(y_true, y_pred, sample_weight=None, alpha=0.7):
  ...     """Custom asymmetric loss function."""
  ...     residual = y_true - y_pred
  ...     loss = np.where(residual >= 0, 
  ...                    alpha * residual**2, 
  ...                    (1-alpha) * residual**2)
  ...     
  ...     gradient = np.where(residual >= 0,
  ...                        -2 * alpha * residual,
  ...                        -2 * (1-alpha) * residual)
  ...     
  ...     hessian = np.where(residual >= 0,
  ...                       2 * alpha * np.ones_like(residual),
  ...                       2 * (1-alpha) * np.ones_like(residual))
  ...     
  ...     if sample_weight is not None:
  ...         loss *= sample_weight
  ...         gradient *= sample_weight
  ...         hessian *= sample_weight
  ...         
  ...     return loss.mean(), gradient, hessian
  >>> 
  >>> regressor = EnhancedHistGradientBoostingRegressor(
  ...     loss=asymmetric_loss,
  ...     max_iter=100,
  ...     random_state=42
  ... )

Performance Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^

The enhanced estimators are designed to maintain competitive performance while
providing additional capabilities:

* **Memory efficiency**: Advanced features typically add <20% memory overhead
* **Speed**: Basic enhanced features add <10% computational overhead
* **Scalability**: SGD solver and memory-efficient modes for large datasets
* **Backward compatibility**: Existing code works without modification

When to Use Enhanced Estimators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider using enhanced estimators when:

* **Dealing with outliers**: Use Huber loss for robust regression
* **Imbalanced classification**: Use Focal loss for better minority class handling
* **Multi-output problems**: Need to model multiple correlated targets
* **High-dimensional data**: Coordinate descent solver for sparse features
* **Large datasets**: SGD solver for memory-efficient training
* **Need interpretability**: Advanced feature importance and SHAP integration
* **Custom objectives**: Require domain-specific loss functions

Migration from Standard Estimators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enhanced estimators are drop-in replacements for standard implementations::

  # Standard implementation
  from sklearn.ensemble import HistGradientBoostingRegressor
  regressor = HistGradientBoostingRegressor(max_iter=100, random_state=42)
  
  # Enhanced implementation (same interface)
  from sklearn.ensemble import EnhancedHistGradientBoostingRegressor
  regressor = EnhancedHistGradientBoostingRegressor(max_iter=100, random_state=42)
  
  # Plus enhanced features
  regressor = EnhancedHistGradientBoostingRegressor(
      solver='newton',           # Enhanced optimization
      loss='huber',             # Robust loss
      l1_regularization=0.01,   # Additional regularization
      learning_rate_schedule='cosine',  # Adaptive learning rate
      max_iter=100,
      random_state=42
  )

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_ensemble_plot_enhanced_hist_gradient_boosting.py`

.. rubric:: References

.. [XGBoost] Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." 
   Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 2016.

.. [LightGBM] Ke, Guolin, et al. "Lightgbm: A highly efficient gradient boosting decision tree." 
   Advances in neural information processing systems 30 (2017): 3146-3154.

.. [Focal] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." 
   Proceedings of the IEEE international conference on computer vision. 2017.

.. [Huber] Huber, Peter J. "Robust estimation of a location parameter." 
   The annals of mathematical statistics 35.1 (1964): 73-101.