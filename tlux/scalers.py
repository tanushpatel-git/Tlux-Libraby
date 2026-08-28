"""Scaling / preprocessing utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

# ---------------------------------------------------------------------------
# Scaler catalogue
# ---------------------------------------------------------------------------

SCALERS: Dict[str, Type[TransformerMixin]] = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
    "MaxAbsScaler": MaxAbsScaler,
    "Normalizer": Normalizer,
    "PowerTransformer": PowerTransformer,
    "QuantileTransformer": QuantileTransformer,
}


def _build_preprocessor(
    scaler_name: Optional[str],
    scale_columns: Optional[List[int]],
    n_features: int,
) -> Optional[ColumnTransformer]:
    """Return a ``ColumnTransformer`` that scales only *scale_columns*.

    If *scaler_name* is ``None`` or ``"No Scaling"``, returns ``None``.
    If *scale_columns* is ``None``, **all** columns are scaled.
    """
    if scaler_name is None or scaler_name == "No Scaling":
        return None

    scaler_cls = SCALERS.get(scaler_name)
    if scaler_cls is None:
        raise ValueError(f"Unknown scaler '{scaler_name}'. Choose from: {list(SCALERS)}")

    if scale_columns is None:
        columns = list(range(n_features))
    else:
        columns = list(scale_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", scaler_cls(), columns),
        ],
        remainder="passthrough",
    )
    return preprocessor


def get_scaler_names(include_no_scaling: bool = True) -> List[str]:
    """Return the list of available scaler names."""
    names = list(SCALERS.keys())
    if include_no_scaling:
        names.insert(0, "No Scaling")
    return names
