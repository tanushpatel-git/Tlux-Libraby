"""Public API — the ``compare_models`` entry point."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .comparator import run_experiments
from .leaderboard import build_leaderboard


def compare_models(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    *,
    is_classification: bool,
    scale: Union[bool, str, List[str]] = False,
    scale_columns: Optional[List[int]] = None,
    models: Optional[Dict[str, Any]] = None,
    scalers: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    problem: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compare many ML models with different scalers and return a leaderboard.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : array-like
        Train/test data.
    is_classification : bool
        **Required.** ``True`` → classification, ``False`` → regression.
        Must be passed explicitly; there is no auto-detection.
    scale : bool | str | list[str]
        Scaling select. ``False`` (default) → runs on the **raw, unscaled**
        data only. ``True`` → tries every scaling technique. A scaler name
        (``"StandardScaler"``) or a list of names
        (``["MinMaxScaler", "RobustScaler"]``) → tries **only** those specific
        techniques.
    scale_columns : list[int] | None
        Column indices to scale. Only active when scaling is enabled. ``None``
        → scale all columns; ``[1, 2, 4]`` → scale only those columns.
    models : str | list | type | dict | None
        Which model(s) to run. Accepts a single name (``"XGBoost"``),
        a list of names (``["XGBoost", "Random Forest"]``), one or more
        estimator classes, or a ``{name: class}`` dict. ``None`` → all
        built-in models for the chosen problem type.
    scalers : list[str] | None
        Scaler names to evaluate. Highest priority: overrides ``scale`` when
        provided. ``None`` → decided by ``scale`` (raw only, all scalers, or
        specific scalers).
    metrics : dict | None
        Custom ``{name: callable}`` mapping. ``None`` → built-in set.
    problem : str | None
        ``"classification"`` or ``"regression"``. Optional cross-check only;
        must agree with ``is_classification``.
    sort_by : str | None
        Metric name to sort the leaderboard by (e.g. ``"F1"``).
        ``None`` → default: sort by ``"Accuracy"`` (classification) or
        ``"R2"`` (regression).
    ascending : bool
        Sort direction. ``False`` = best score on top (default).
    n_jobs : int
        Parallel jobs via joblib (default 1 = sequential).

    Returns
    -------
    pd.DataFrame
        A leaderboard DataFrame sorted by the chosen metric (descending).
    """
    results = run_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        is_classification=is_classification,
        scale=scale,
        scale_columns=scale_columns,
        models=models,
        scalers=scalers,
        metrics=metrics,
        problem=problem,
        n_jobs=n_jobs,
    )

    return build_leaderboard(results, sort_by=sort_by, ascending=ascending)
