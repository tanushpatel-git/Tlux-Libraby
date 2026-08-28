"""Catalogues of classification and regression estimators."""

from __future__ import annotations

from typing import Dict, Type

from sklearn.base import BaseEstimator

# ---------------------------------------------------------------------------
# Optional third-party models (available only if the package is installed)
# ---------------------------------------------------------------------------

HAS_XGBOOST = False
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

# ---------------------------------------------------------------------------
# Classification models
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS: Dict[str, Type[BaseEstimator]] = {}

def _register_classification() -> None:
    from sklearn.linear_model import (
        LogisticRegression,
        SGDClassifier,
        Perceptron,
        PassiveAggressiveClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        AdaBoostClassifier,
    )
    from sklearn.neighbors import (
        KNeighborsClassifier,
        RadiusNeighborsClassifier,
    )
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.naive_bayes import (
        GaussianNB,
        MultinomialNB,
        BernoulliNB,
        ComplementNB,
    )
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )

    CLASSIFICATION_MODELS.update({
        "Logistic Regression": LogisticRegression,
        "SGD Classifier": SGDClassifier,
        "Perceptron": Perceptron,
        "Passive Aggressive Classifier": PassiveAggressiveClassifier,
        "Decision Tree": DecisionTreeClassifier,
        "Random Forest": RandomForestClassifier,
        "Extra Trees": ExtraTreesClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "Hist Gradient Boosting": HistGradientBoostingClassifier,
        "KNN": KNeighborsClassifier,
        "SVC": SVC,
        "Linear SVC": LinearSVC,
        "Nu SVC": NuSVC,
        "Gaussian Naive Bayes": GaussianNB,
        "Multinomial Naive Bayes": MultinomialNB,
        "Bernoulli Naive Bayes": BernoulliNB,
        "Complement Naive Bayes": ComplementNB,
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis,
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis,
        "AdaBoost": AdaBoostClassifier,
    })

    if HAS_XGBOOST:
        CLASSIFICATION_MODELS["XGBoost"] = XGBClassifier

_register_classification()

# ---------------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------------

REGRESSION_MODELS: Dict[str, Type[BaseEstimator]] = {}

def _register_regression() -> None:
    from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
        BayesianRidge,
        HuberRegressor,
        QuantileRegressor,
        SGDRegressor,
        PassiveAggressiveRegressor,
    )
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        AdaBoostRegressor,
    )
    from sklearn.neighbors import (
        KNeighborsRegressor,
        RadiusNeighborsRegressor,
    )
    from sklearn.svm import SVR, LinearSVR, NuSVR

    REGRESSION_MODELS.update({
        "Linear Regression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "Bayesian Ridge": BayesianRidge,
        "Huber Regressor": HuberRegressor,
        "Quantile Regressor": QuantileRegressor,
        "SGD Regressor": SGDRegressor,
        "Passive Aggressive Regressor": PassiveAggressiveRegressor,
        "Decision Tree": DecisionTreeRegressor,
        "Random Forest": RandomForestRegressor,
        "Extra Trees": ExtraTreesRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
        "Hist Gradient Boosting": HistGradientBoostingRegressor,
        "KNN": KNeighborsRegressor,
        "Radius Neighbors": RadiusNeighborsRegressor,
        "SVR": SVR,
        "Linear SVR": LinearSVR,
        "Nu SVR": NuSVR,
        "AdaBoost Regressor": AdaBoostRegressor,
    })

    if HAS_XGBOOST:
        REGRESSION_MODELS["XGBoost"] = XGBRegressor

_register_regression()


def get_model_names(problem: str) -> list:
    """Return the available model names for a problem type."""
    source = CLASSIFICATION_MODELS if problem == "classification" else REGRESSION_MODELS
    return list(source.keys())


def resolve_models(
    models: "Any",
    problem: str,
) -> Dict[str, Type[BaseEstimator]]:
    """Normalise the ``models`` argument into a ``{name: estimator_class}`` dict.

    Accepts any of:
    * ``None``                          → built-in catalogue for *problem*
    * a single name (``str``)           → e.g. ``"XGBoost"``
    * a list of names                   → e.g. ``["XGBoost", "Random Forest"]``
    * an estimator class                → a single custom/known estimator
    * a list of estimator classes
    * a dict ``{name: estimator_class}`` → used as-is

    Names are matched case-insensitively against the built-in catalogue,
    falling back to the class name for custom estimators.
    """
    catalogue = CLASSIFICATION_MODELS if problem == "classification" else REGRESSION_MODELS

    if models is None:
        return dict(catalogue)

    # dict {name: class} -> use as-is
    if isinstance(models, dict):
        return dict(models)

    # single estimator class
    if isinstance(models, type) and issubclass(models, BaseEstimator):
        return {models.__name__: models}

    # normalise str / list into a list of specifiers
    items = [models] if isinstance(models, str) or not isinstance(models, (list, tuple)) else list(models)

    resolved: Dict[str, Type[BaseEstimator]] = {}
    for item in items:
        # string name
        if isinstance(item, str):
            match = next(
                (k for k in catalogue if k.lower() == item.lower()),
                None,
            )
            if match is not None:
                resolved[match] = catalogue[match]
            else:
                raise ValueError(
                    f"Unknown model '{item}'. Available: {', '.join(catalogue)}"
                )
        # estimator class
        elif isinstance(item, type) and issubclass(item, BaseEstimator):
            resolved[item.__name__] = item
        else:
            raise TypeError(
                f"Unsupported model spec: {item!r}. Use a name, an estimator class, "
                "a list of those, or a {name: class} dict."
            )

    return resolved
