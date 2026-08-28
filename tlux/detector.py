"""Problem-type detection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_classification(y: np.ndarray | pd.Series) -> bool:
    """Heuristically decide whether *y* looks like a classification target.

    Returns ``True`` when:
    - *y* dtype is object / bool / category, **or**
    - *y* is numeric but has <= 20 unique values **and** those values are all
      integers.
    """
    arr = np.asarray(y)

    if arr.dtype.kind in ("U", "S", "b", "O", "category"):
        return True

    if arr.dtype.kind == "category":
        return True

    if arr.dtype.kind in ("f", "i", "u"):
        n_unique = len(np.unique(arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr))
        if n_unique <= 20 and np.all(arr[~np.isnan(arr)] == arr[~np.isnan(arr)].astype(int)) if arr.dtype.kind == "f" else n_unique <= 20 and np.all(arr == arr.astype(int)):
            return True

    return False
