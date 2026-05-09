# moirais.fn — function file (hadesllm/moirais)
"""Proportions per group with confidence interval."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import DescriptiveResult


def otis_demo_proportion(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
    total: int | None = None,
) -> DescriptiveResult:
    """Group proportions with Wilson confidence intervals.

    Parameters
    ----------
    df : DataFrame
    group_col : str
    total : int, optional
        Denominator. If None, uses len(df).

    Returns
    -------
    DescriptiveResult
    """
    n = total if total is not None else len(df)
    counts = df[group_col].value_counts()
    z = 1.96
    results = {}
    for g, c in counts.items():
        p = c / max(n, 1)
        denom = 1 + z**2 / max(n, 1)
        center = p + z**2 / (2 * max(n, 1))
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * max(n, 1))) / max(n, 1))
        ci_lo = max((center - margin) / denom, 0)
        ci_hi = min((center + margin) / denom, 1)
        results[str(g)] = {"proportion": float(p), "ci_lower": float(ci_lo), "ci_upper": float(ci_hi), "count": int(c)}
    return DescriptiveResult(name="otis_demo_proportion", value=len(results), extra={"proportions": results, "n": n})


odm_p = otis_demo_proportion


def cheatsheet() -> str:
    return "otis_demo_proportion({}) -> Proportions per group with confidence interval."
