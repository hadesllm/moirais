# morie.fn — function file (hadesllm/morie)
"""Cyclist safety analysis."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def mto_cyclist(
    df: pd.DataFrame,
    *,
    cyclist_col: str = "cyclist_involved",
    severity_col: str = "severity",
) -> DescriptiveResult:
    """Analyse cyclist collision characteristics.

    Parameters
    ----------
    df : DataFrame
    cyclist_col : str
    severity_col : str

    Returns
    -------
    DescriptiveResult
    """
    if cyclist_col not in df.columns:
        raise ValueError(f"Column '{cyclist_col}' not found")
    cyc = df[df[cyclist_col].astype(bool)]
    n_cyc = len(cyc)
    extra = {"n_cyclist": n_cyc, "n_total": len(df), "pct_cyclist": n_cyc / len(df) if len(df) > 0 else 0.0}
    if severity_col in cyc.columns:
        extra["severity_dist"] = cyc[severity_col].value_counts().to_dict()
    return DescriptiveResult(name="cyclist_collision", value=float(n_cyc), extra=extra)


mtocyc = mto_cyclist


def cheatsheet() -> str:
    return "mto_cyclist({}) -> Cyclist safety analysis."
