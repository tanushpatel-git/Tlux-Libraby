"""Core comparison engine — trains every (model × scaler) combination."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .detector import is_classification as _detect_classification
from .models import CLASSIFICATION_MODELS, REGRESSION_MODELS, resolve_models
from .scalers import SCALERS, _build_preprocessor, get_scaler_names

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

CLASSIFICATION_METRICS = {
    "Accuracy": accuracy_score,
    "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted", zero_division=0),
    "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted", zero_division=0),
    "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0),
}

REGRESSION_METRICS = {
    "R2": r2_score,
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "RMSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred) ** 0.5,
}


def _get_metrics(problem: str) -> Dict[str, Any]:
    return CLASSIFICATION_METRICS if problem == "classification" else REGRESSION_METRICS


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def _run_single_experiment(
    model_cls: Any,
    model_name: str,
    scaler_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_columns: Optional[List[int]],
    metrics_fn: Dict[str, Any],
) -> Dict[str, Any]:
    """Train one model with one scaler and return the result dict."""
    result: Dict[str, Any] = {
        "model_name": model_name,
        "scaler_name": scaler_name,
        "scaled_columns_str": ", ".join(map(str, scale_columns)) if scale_columns else "—",
        "metrics": {},
        "error": None,
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            preprocessor = _build_preprocessor(scaler_name, scale_columns, X_train.shape[1])

            steps = []
            if preprocessor is not None:
                steps.append(("preprocessor", preprocessor))

            model = model_cls()
            steps.append(("model", model))

            pipeline = Pipeline(steps)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            for metric_name, metric_fn in metrics_fn.items():
                try:
                    score = metric_fn(y_test, y_pred)
                    result["metrics"][metric_name] = round(float(score), 4)
                except Exception as exc:
                    result["metrics"][metric_name] = f"err: {exc}"

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def run_experiments(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    *,
    is_classification: bool,
    scale: Union[bool, str, List[str]] = False,
    scale_columns: Optional[List[int]] = None,
    models: Optional[Dict[str, Any]] = None,
    scalers: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    problem: Optional[str] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run all (model × scaler) experiments and return result dicts.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : array-like
        Train/test splits.
    is_classification : bool
        **Required.** ``True`` → run classification models, ``False`` → run
        regression models. The user must choose explicitly, and it is validated
        against ``y_train`` (a mismatch raises ``ValueError``).
    scale : bool | str | list[str]
        Scaling select. ``False`` (default) → run on the **raw, unscaled** data
        only (a single ``"No Scaling"`` entry). ``True`` → try every scaling
        technique. A scaler name (``"StandardScaler"``) or a list of names
        (``["MinMaxScaler", "RobustScaler"]``) → try **only** those specific
        techniques.
    scale_columns : list[int] | None
        Column indices to scale. Only active when scaling is enabled. ``None``
        means *all* columns are scaled; a list like ``[1, 2, 4]`` scales only
        those columns.
    models : str | list | type | dict | None
        Which model(s) to run. Accepts a single name (``"XGBoost"``),
        a list of names (``["XGBoost", "Random Forest"]``), one or more
        estimator classes, or a ``{name: class}`` dict. ``None`` → all
        built-in models for the chosen problem type.
    scalers : list[str] | None
        Exact scaler names to try. Highest priority: when provided it overrides
        the ``scale`` switch. ``None`` → decided by ``scale`` (raw only, or all
        techniques).
    metrics : dict | None
        ``{name: callable}`` mapping. ``None`` uses the built-in set.
    problem : str | None
        ``"classification"`` or ``"regression"``. Used only as a cross-check
        against ``is_classification``; not used for auto-detection.
    n_jobs : int
        Number of parallel jobs (via joblib). 1 = sequential.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    list[dict]
        One dict per experiment.

    Raises
    ------
    TypeError
        If ``is_classification`` is omitted or not a bool.
    ValueError
        If ``is_classification`` conflicts with an explicit ``problem``, or if
        ``is_classification`` does not match the detected type of ``y_train``
        (no auto-detection — a mismatch raises instead of running).
    """
    if not isinstance(is_classification, bool):
        raise TypeError(
            "is_classification is required and must be a bool "
            "(True for classification, False for regression)."
        )

    X_train = np.asarray(X_train) if isinstance(X_train, pd.DataFrame) else X_train
    X_test = np.asarray(X_test) if isinstance(X_test, pd.DataFrame) else X_test
    y_train = np.asarray(y_train) if isinstance(y_train, pd.Series) else y_train
    y_test = np.asarray(y_test) if isinstance(y_test, pd.Series) else y_test

    detected_classification = _detect_classification(y_train)
    if is_classification and not detected_classification:
        raise ValueError(
            "is_classification=True, but y_train does not look like a "
            "classification target (continuous/float values, or too many unique "
            "values). Pass is_classification=False for regression."
        )
    if not is_classification and detected_classification:
        raise ValueError(
            "is_classification=False, but y_train looks like a classification "
            "target (categorical, or few unique integer values). Pass "
            "is_classification=True instead."
        )

    if problem is not None:
        if problem not in ("classification", "regression"):
            raise ValueError(
                f"problem must be 'classification' or 'regression', got {problem!r}"
            )
        expected = "classification" if is_classification else "regression"
        if problem != expected:
            raise ValueError(
                f"Conflict: is_classification={is_classification} implies "
                f"'{expected}', but problem={problem!r}."
            )

    problem = "classification" if is_classification else "regression"
    if verbose:
        print(f"Problem type: {problem}")

    if models is None:
        models = CLASSIFICATION_MODELS if problem == "classification" else REGRESSION_MODELS

    models = resolve_models(models, problem)

    if scalers is None:
        if isinstance(scale, str):
            scalers = [scale]
        elif isinstance(scale, (list, tuple)):
            scalers = list(scale)
        elif scale is True:
            scalers = get_scaler_names(include_no_scaling=True)
        elif scale is False:
            scalers = ["No Scaling"]
        else:
            raise TypeError(
                f"scale must be a bool, a scaler name, or a list of scaler "
                f"names; got {scale!r}."
            )

        valid = set(get_scaler_names(include_no_scaling=True))
        unknown = [s for s in scalers if s not in valid]
        if unknown:
            raise ValueError(
                f"Unknown scaler(s): {unknown}. Available: {', '.join(sorted(valid))}"
            )

    if metrics is None:
        metrics = _get_metrics(problem)

    total = len(models) * len(scalers)
    if verbose:
        print(f"Running {total} experiments ({len(models)} models × {len(scalers)} scalers)...\n")

    combos = [
        (model_cls, model_name, scaler_name)
        for model_name, model_cls in models.items()
        for scaler_name in scalers
    ]

    if n_jobs == 1:
        results = [
            _run_single_experiment(
                mc, mn, sn, X_train, y_train, X_test, y_test, scale_columns, metrics
            )
            for mc, mn, sn in combos
        ]
    else:
        jobs = [
            delayed(_run_single_experiment)(
                mc, mn, sn, X_train, y_train, X_test, y_test, scale_columns, metrics
            )
            for mc, mn, sn in combos
        ]
        results = Parallel(n_jobs=n_jobs)(jobs)

    succeeded = sum(1 for r in results if r["error"] is None)
    if verbose:
        print(f"Completed: {succeeded}/{total} succeeded.\n")

    return results
