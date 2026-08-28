"""Leaderboard formatting and display."""

from __future__ import annotations

from typing import Any, Dict, List

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


def print_leaderboard(df: pd.DataFrame, sort_by: str | None = None) -> None:
    """Pretty-print the leaderboard to stdout."""
    if df.empty:
        print("No results to display.")
        return

    print("\n" + "=" * 80)
    header = "  MODEL LEADERBOARD"
    if sort_by:
        header += f"  (sorted by {sort_by})"
    print(header)
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


def print_experiment_detail(result: Dict[str, Any]) -> None:
    """Print a detailed breakdown of a single experiment."""
    print(f"\n--- Experiment: {result['model_name']} ---")
    print(f"  Preprocessing : {result['scaler_name']}")
    print(f"  Scaled Columns: {result.get('scaled_columns_str', '—')}")
    if result.get("error"):
        print(f"  ERROR         : {result['error']}")
    else:
        for metric_name, value in result.get("metrics", {}).items():
            print(f"  {metric_name:<15}: {value}")
    print()
