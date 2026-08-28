"""Leaderboard formatting and display."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def build_leaderboard(
    results: List[Dict[str, Any]],
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """Convert a list of experiment dicts into a sorted ``DataFrame``.

    Parameters
    ----------
    results : list[dict]
        Experiment result dicts.
    sort_by : str | None
        Metric column name to sort by (e.g. ``"F1"``). ``None`` → default:
        sort by ``"Accuracy"`` (classification) or ``"R2"`` (regression).
    ascending : bool
        Sort direction. ``False`` = best score on top (default).

    Returns
    -------
    pd.DataFrame
        Sorted leaderboard.
    """
    if not results:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for r in results:
        row: Dict[str, Any] = {
            "Model": r["model_name"],
            "Scaling": r["scaler_name"],
            "Scaled Columns": r.get("scaled_columns_str", "—"),
        }
        for metric_name, value in r.get("metrics", {}).items():
            row[metric_name] = value
        rows.append(row)

    df = pd.DataFrame(rows)

    metric_cols = [c for c in df.columns if c not in ("Model", "Scaling", "Scaled Columns")]
    if metric_cols:
        if sort_by is not None:
            if sort_by not in metric_cols:
                available = ", ".join(metric_cols)
                raise ValueError(
                    f"sort_by='{sort_by}' not found. Available metrics: {available}"
                )
            sort_key = sort_by
        else:
            if "Accuracy" in metric_cols:
                sort_key = "Accuracy"
            elif "R2" in metric_cols:
                sort_key = "R2"
            else:
                sort_key = metric_cols[0]

        df["_sort"] = pd.to_numeric(df[sort_key], errors="coerce")
        df = (
            df.sort_values(by="_sort", ascending=ascending, na_position="last")
            .drop(columns=["_sort"])
            .reset_index(drop=True)
        )

    return df
