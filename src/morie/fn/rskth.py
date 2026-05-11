# morie.fn — function file (hadesllm/morie)
"""Optimal risk classification threshold via Youden's J."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def risk_threshold(
    df: pd.DataFrame,
    *,
    score_col: str = DEFAULT_COLS["outcome"],
    outcome_col: str = DEFAULT_COLS["treatment"],
) -> dict:
    """Find optimal classification threshold using Youden's J statistic.

    J = sensitivity + specificity - 1. The threshold maximizing J is
    returned along with operating characteristics.

    Parameters
    ----------
    df : DataFrame
        Dataset with score and binary outcome columns.
    score_col : str
        Column with continuous risk score.
    outcome_col : str
        Column with binary outcome (1 = event).

    Returns
    -------
    dict
        threshold, sensitivity, specificity, ppv, npv, youden_j.
    """
    tmp = df[[score_col, outcome_col]].dropna()
    scores = tmp[score_col].values.astype(float)
    labels = tmp[outcome_col].values.astype(int)

    thresholds = np.unique(scores)
    if len(thresholds) < 2:
        return {
            "threshold": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "ppv": np.nan,
            "npv": np.nan,
            "youden_j": np.nan,
        }

    best_j = -1.0
    best = {}

    for t in thresholds:
        pred = (scores > t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        j = sens + spec - 1.0

        if j > best_j:
            best_j = j
            best = {
                "threshold": float(t),
                "sensitivity": sens,
                "specificity": spec,
                "ppv": ppv,
                "npv": npv,
                "youden_j": j,
            }

    return best


rskth = risk_threshold


def cheatsheet() -> str:
    return "risk_threshold({}) -> Optimal risk classification threshold via Youden's J."
