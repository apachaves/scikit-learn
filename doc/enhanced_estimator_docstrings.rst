Enhanced HistGradientBoosting Estimator Docstrings
===================================================

This file contains the comprehensive docstring templates for the Enhanced HistGradientBoosting
estimators following scikit-learn documentation conventions.

EnhancedHistGradientBoostingRegressor Docstring
-----------------------------------------------

.. code-block:: python

    """Enhanced Histogram-based Gradient Boosting Regression Tree.
    
    This estimator extends the standard HistGradientBoostingRegressor with modern
    machine learning techniques, including multiple optimization solvers, robust
    loss functions, advanced regularization, and enhanced interpretability features.
    
    The enhanced estimator maintains full backward compatibility with the standard
    implementation while providing significant improvements for challenging datasets,
    including better handling of outliers, multi-output scenarios, and high-dimensional
    sparse data.
    
    Read more in the :ref:`User Guide <enhanced_histogram_based_gradient_boosting>`.
    
    .. versionadded:: 1.7
        Enhanced HistGradientBoosting estimators with modern ML techniques.
    
    Parameters
    ----------
    loss : {'squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile', 'huber'} \
            or callable, default='squared_error'
        The loss function to use in the boosting process. Enhanced version supports
        additional robust loss functions:
        
        - 'huber': Robust loss function that is quadratic for small residuals and
          linear for large residuals, making it less sensitive to outliers.
        - Custom callable: User-defined loss function following the gradient boosting
          protocol. Must return (loss, gradient, hessian).
    
    solver : {'standard', 'newton', 'sgd', 'coordinate'}, default='standard'
        The optimization solver to use for gradient boosting:
        
        - 'standard': Uses the standard gradient descent approach.
        - 'newton': Enhanced Newton-Raphson solver with adaptive step sizing.
        - 'sgd': Stochastic gradient descent with momentum and mini-batch support.
        - 'coordinate': Coordinate descent solver for high-dimensional sparse data.
    
    learning_rate : float, default=0.1
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1.0`` for no shrinkage.
    
    max_iter : int, default=100
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees for binary classification. For multiclass
        classification, `n_classes` trees per iteration are built.
    
    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.
    
    max_depth : int or None, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
    
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.
    
    l2_regularization : float, default=0.0
        The L2 regularization parameter penalizing leaves with small hessians.
        Use values in [0, inf).
    
    l1_regularization : float, default=0.0
        The L1 regularization parameter promoting sparsity in leaf weights.
        Use values in [0, inf). Enhanced feature for feature selection.
    
    max_features : float, default=1.0
        Proportion of randomly chosen features in each and every node split.
        This is a form of regularization, smaller values make the trees weaker
        learners and might help preventing overfitting.
    
    max_bins : int, default=255
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.
    
    categorical_features : array-like of {bool, int, str} of shape (n_features,) \
            or None, default=None
        Indicates the categorical features.
    
        - None : no feature will be considered categorical.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical features.
        - str array-like: names of categorical features (assuming the training
          data has feature names).
    
    monotonic_cst : array-like of int of shape (n_features,), default=None
        Indicates the monotonic constraint to enforce on each feature.
        - 1: monotonic increase
        - 0: no constraint
        - -1: monotonic decrease
    
    interaction_cst : {"pairwise", "no_interactions"} or sequence of lists/tuples/sets \
            of int, default=None
        Specify interaction constraints, the sets of features which can
        interact with each other in child node splits.
    
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble. For results to be valid, the
        estimator should be re-trained on the same data only.
    
    early_stopping : 'auto' or bool, default='auto'
        If 'auto', early stopping is enabled if the sample size is larger than
        10000. If True, early stopping is enabled, otherwise early stopping is
        disabled.
    
    scoring : str or callable or None, default='loss'
        Scoring parameter to use for early stopping. It can be a single
        string or a callable.
    
    validation_fraction : int or float or None, default=0.1
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping.
    
    n_iter_no_change : int, default=10
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1`` -th-to-last one, up to some
        tolerance.
    
    tol : float, default=1e-7
        The absolute tolerance to use when comparing scores. The higher the
        tolerance, the more likely we are to early stop: higher tolerance
        means that it will be harder for subsequent iterations to be
        considered an improvement upon the reference score.
    
    verbose : int, default=0
        The verbosity level. If not zero, print some information about the
        fitting process.
    
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled.
    
    learning_rate_schedule : {'constant', 'cosine', 'exponential', 'step', 'adaptive'}, \
            default='constant'
        Learning rate scheduling strategy. Enhanced feature for adaptive learning:
        
        - 'constant': Fixed learning rate throughout training.
        - 'cosine': Cosine annealing schedule.
        - 'exponential': Exponential decay schedule.
        - 'step': Step-wise decay at fixed intervals.
        - 'adaptive': Adaptive learning rate based on validation performance.
    
    multi_output : bool, default=False
        Whether to enable multi-output regression capabilities. Enhanced feature
        for handling multiple correlated target variables simultaneously.
    
    bagging : bool, default=False
        Whether to enable bootstrap aggregating for ensemble diversity.
        Enhanced feature for improved generalization.
    
    outlier_detection : bool, default=False
        Whether to enable automatic outlier detection and handling.
        Enhanced feature for robust learning.
    
    enhanced_config : EnhancedBoostingConfig or None, default=None
        Advanced configuration object for enhanced features including
        regularization parameters, ensemble diversity settings, and
        solver-specific parameters.
    
    Attributes
    ----------
    n_iter_ : int
        The number of iterations as selected by early stopping, depending on
        the `early_stopping` parameter. Otherwise it corresponds to max_iter.
    
    n_trees_per_iteration_ : int
        The number of tree that are built at each iteration. This is equal to 1
        for binary classification and regression, and to ``n_classes`` for
        multiclass classification.
    
    train_score_ : ndarray, shape (n_iter_,)
        The scores at each iteration on the training data. The first entry
        is the score of the ensemble before the first iteration. Scores are
        computed according to the ``scoring`` parameter. If ``scoring`` is
        not 'loss', scores are computed on a subset of at most 10 000
        samples. Empty if no early stopping.
    
    validation_score_ : ndarray, shape (n_iter_,)
        The scores at each iteration on the validation data. The first entry
        is the score of the ensemble before the first iteration. Scores are
        computed according to the ``scoring`` parameter. Empty if no early
        stopping or if ``validation_fraction`` is None.
    
    is_categorical_ : ndarray, shape (n_features,) or None
        Boolean mask for the categorical features. ``None`` if there are no
        categorical features.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,), dtype=object
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
    See Also
    --------
    HistGradientBoostingRegressor : Standard histogram-based gradient boosting regressor.
    EnhancedHistGradientBoostingClassifier : Enhanced histogram-based gradient boosting classifier.
    GradientBoostingRegressor : Exact gradient boosting regressor.
    
    Notes
    -----
    The enhanced estimator provides all capabilities of the standard
    HistGradientBoostingRegressor plus additional modern ML techniques:
    
    - **Multiple Solvers**: Newton-Raphson, SGD, and Coordinate Descent solvers
      for different optimization scenarios.
    - **Robust Loss Functions**: Huber loss for outlier-robust regression.
    - **Advanced Regularization**: L1 regularization, Elastic Net, and Dropout.
    - **Multi-output Support**: Handle multiple correlated targets simultaneously.
    - **Enhanced Interpretability**: SHAP integration and advanced feature importance.
    
    The features are designed to maintain backward compatibility while providing
    significant improvements for challenging datasets.
    
    References
    ----------
    .. [1] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System",
           22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016.
    .. [2] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
           Advances in Neural Information Processing Systems 30, 2017.
    .. [3] P. J. Huber, "Robust Estimation of a Location Parameter",
           The Annals of Mathematical Statistics, 1964.
    
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import EnhancedHistGradientBoostingRegressor
    >>> X, y = make_regression(n_samples=500, random_state=0)
    >>> reg = EnhancedHistGradientBoostingRegressor(random_state=0).fit(X, y)
    >>> reg.score(X, y)
    0.99...
    
    Enhanced features example:
    
    >>> # Robust regression with outliers
    >>> reg_robust = EnhancedHistGradientBoostingRegressor(
    ...     loss='huber',
    ...     solver='newton',
    ...     l1_regularization=0.01,
    ...     random_state=0
    ... )
    >>> reg_robust.fit(X, y)
    EnhancedHistGradientBoostingRegressor(...)
    
    >>> # Multi-output regression
    >>> import numpy as np
    >>> y_multi = np.column_stack([y, y * 0.5])
    >>> reg_multi = EnhancedHistGradientBoostingRegressor(
    ...     multi_output=True,
    ...     random_state=0
    ... )
    >>> reg_multi.fit(X, y_multi)
    EnhancedHistGradientBoostingRegressor(...)
    >>> reg_multi.predict(X[:5])  # doctest: +SKIP
    array([...])
    
    >>> # Enhanced feature importance
    >>> importance = reg.get_feature_importance(method='gain')
    >>> importance.shape
    (100,)
    """

EnhancedHistGradientBoostingClassifier Docstring
------------------------------------------------

.. code-block:: python

    """Enhanced Histogram-based Gradient Boosting Classification Tree.
    
    This estimator extends the standard HistGradientBoostingClassifier with modern
    machine learning techniques, including multiple optimization solvers, robust
    loss functions (like Focal loss for imbalanced data), advanced regularization,
    and enhanced interpretability features.
    
    The enhanced estimator maintains full backward compatibility with the standard
    implementation while providing significant improvements for challenging datasets,
    including better handling of imbalanced classes and high-dimensional sparse data.
    
    Read more in the :ref:`User Guide <enhanced_histogram_based_gradient_boosting>`.
    
    .. versionadded:: 1.7
        Enhanced HistGradientBoosting estimators with modern ML techniques.
    
    Parameters
    ----------
    loss : {'log_loss', 'focal'} or callable, default='log_loss'
        The loss function to use in the boosting process. Enhanced version supports
        additional robust loss functions:
        
        - 'focal': Focal loss designed for imbalanced classification problems.
          Down-weights easy examples and focuses learning on hard examples.
        - Custom callable: User-defined loss function following the gradient boosting
          protocol. Must return (loss, gradient, hessian).
    
    solver : {'standard', 'newton', 'sgd', 'coordinate'}, default='standard'
        The optimization solver to use for gradient boosting:
        
        - 'standard': Uses the standard gradient descent approach.
        - 'newton': Enhanced Newton-Raphson solver with adaptive step sizing.
        - 'sgd': Stochastic gradient descent with momentum and mini-batch support.
        - 'coordinate': Coordinate descent solver for high-dimensional sparse data.
    
    learning_rate : float, default=0.1
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1.0`` for no shrinkage.
    
    max_iter : int, default=100
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees for binary classification. For multiclass
        classification, `n_classes` trees per iteration are built.
    
    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.
    
    max_depth : int or None, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
    
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.
    
    l2_regularization : float, default=0.0
        The L2 regularization parameter penalizing leaves with small hessians.
        Use values in [0, inf).
    
    l1_regularization : float, default=0.0
        The L1 regularization parameter promoting sparsity in leaf weights.
        Use values in [0, inf). Enhanced feature for feature selection.
    
    max_features : float, default=1.0
        Proportion of randomly chosen features in each and every node split.
        This is a form of regularization, smaller values make the trees weaker
        learners and might help preventing overfitting.
    
    max_bins : int, default=255
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.
    
    categorical_features : array-like of {bool, int, str} of shape (n_features,) \
            or None, default=None
        Indicates the categorical features.
    
    monotonic_cst : array-like of int of shape (n_features,), default=None
        Indicates the monotonic constraint to enforce on each feature.
        - 1: monotonic increase
        - 0: no constraint
        - -1: monotonic decrease
    
    interaction_cst : {"pairwise", "no_interactions"} or sequence of lists/tuples/sets \
            of int, default=None
        Specify interaction constraints, the sets of features which can
        interact with each other in child node splits.
    
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble. For results to be valid, the
        estimator should be re-trained on the same data only.
    
    early_stopping : 'auto' or bool, default='auto'
        If 'auto', early stopping is enabled if the sample size is larger than
        10000. If True, early stopping is enabled, otherwise early stopping is
        disabled.
    
    scoring : str or callable or None, default='loss'
        Scoring parameter to use for early stopping.
    
    validation_fraction : int or float or None, default=0.1
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping.
    
    n_iter_no_change : int, default=10
        Used to determine when to "early stop".
    
    tol : float, default=1e-7
        The absolute tolerance to use when comparing scores during early stopping.
    
    verbose : int, default=0
        The verbosity level. If not zero, print some information about the
        fitting process.
    
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled.
    
    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
    
    learning_rate_schedule : {'constant', 'cosine', 'exponential', 'step', 'adaptive'}, \
            default='constant'
        Learning rate scheduling strategy. Enhanced feature for adaptive learning.
    
    bagging : bool, default=False
        Whether to enable bootstrap aggregating for ensemble diversity.
        Enhanced feature for improved generalization.
    
    outlier_detection : bool, default=False
        Whether to enable automatic outlier detection and handling.
        Enhanced feature for robust learning.
    
    enhanced_config : EnhancedBoostingConfig or None, default=None
        Advanced configuration object for enhanced features.
    
    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        Class labels.
    
    n_iter_ : int
        The number of iterations as selected by early stopping.
    
    n_trees_per_iteration_ : int
        The number of tree that are built at each iteration.
    
    train_score_ : ndarray, shape (n_iter_,)
        The scores at each iteration on the training data.
    
    validation_score_ : ndarray, shape (n_iter_,)
        The scores at each iteration on the validation data.
    
    is_categorical_ : ndarray, shape (n_features,) or None
        Boolean mask for the categorical features.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,), dtype=object
        Names of features seen during :term:`fit`.
    
    See Also
    --------
    HistGradientBoostingClassifier : Standard histogram-based gradient boosting classifier.
    EnhancedHistGradientBoostingRegressor : Enhanced histogram-based gradient boosting regressor.
    GradientBoostingClassifier : Exact gradient boosting classifier.
    
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import EnhancedHistGradientBoostingClassifier
    >>> X, y = make_classification(n_samples=500, random_state=0)
    >>> clf = EnhancedHistGradientBoostingClassifier(random_state=0).fit(X, y)
    >>> clf.score(X, y)
    1.0
    
    Enhanced features for imbalanced data:
    
    >>> # Focal loss for imbalanced classification
    >>> X_imb, y_imb = make_classification(
    ...     n_samples=1000, weights=[0.9, 0.1], random_state=0
    ... )
    >>> clf_focal = EnhancedHistGradientBoostingClassifier(
    ...     loss='focal',
    ...     solver='sgd',
    ...     random_state=0
    ... )
    >>> clf_focal.fit(X_imb, y_imb)
    EnhancedHistGradientBoostingClassifier(...)
    """