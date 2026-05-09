# moirais.fn — function file (hadesllm/moirais)
"""Pedestrian collision analysis."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def mto_pedestrian(
    df: pd.DataFrame,
    *,
    pedestrian_col: str = "pedestrian_involved",
    severity_col: str = "severity",
) -> DescriptiveResult:
    """Analyse pedestrian collision characteristics.

    Parameters
    ----------
    df : DataFrame
    pedestrian_col : str
        Boolean or 0/1 indicator.
    severity_col : str

    Returns
    -------
    DescriptiveResult
    """
    if pedestrian_col not in df.columns:
        raise ValueError(f"Column '{pedestrian_col}' not found")
    ped = df[df[pedestrian_col].astype(bool)]
    n_ped = len(ped)
    n_total = len(df)
    extra = {"n_pedestrian": n_ped, "n_total": n_total, "pct_pedestrian": n_ped / n_total if n_total > 0 else 0.0}
    if severity_col in ped.columns:
        extra["severity_dist"] = ped[severity_col].value_counts().to_dict()
    return DescriptiveResult(name="pedestrian_collision", value=float(n_ped), extra=extra)


mtoped = mto_pedestrian


def cheatsheet() -> str:
    return "mto_pedestrian({}) -> Pedestrian collision analysis."
