# morie.fn — function file (hadesllm/morie)
"""Quasi-experimental design analysis (matching + DiD)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import ESRes


def program_quasi_exp(
    df: pd.DataFrame,
    *,
    outcome_col: str = "outcome",
    treatment_col: str = "treatment",
    time_col: str = "post",
) -> ESRes:
    """Estimate program effect via difference-in-differences.

    Parameters
    ----------
    df : DataFrame
    outcome_col : str
    treatment_col : str
        Binary treatment group indicator.
    time_col : str
        Binary pre/post indicator (0 = pre, 1 = post).

    Returns
    -------
    ESRes
    """
    for c in [outcome_col, treatment_col, time_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found")

    y = df[outcome_col].values.astype(float)
    d = df[treatment_col].values.astype(float)
    t = df[time_col].values.astype(float)

    y_t1_post = y[(d == 1) & (t == 1)]
    y_t1_pre = y[(d == 1) & (t == 0)]
    y_t0_post = y[(d == 0) & (t == 1)]
    y_t0_pre = y[(d == 0) & (t == 0)]

    if any(len(a) == 0 for a in [y_t1_post, y_t1_pre, y_t0_post, y_t0_pre]):
        raise ValueError("All four cells (treatment x time) must have observations")

    did = (np.mean(y_t1_post) - np.mean(y_t1_pre)) - (np.mean(y_t0_post) - np.mean(y_t0_pre))
    n = len(y)
    se = float(np.std(y, ddof=1) * np.sqrt(4 / n))
    return ESRes(
        measure="did_estimate",
        estimate=float(did),
        ci_lower=float(did - 1.96 * se),
        ci_upper=float(did + 1.96 * se),
        se=se,
        n=n,
    )


prgqe = program_quasi_exp


def cheatsheet() -> str:
    return "program_quasi_exp({}) -> Quasi-experimental design analysis (matching + DiD)."
