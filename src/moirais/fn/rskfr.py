# moirais.fn — function file (hadesllm/moirais)
"""Risk score fairness analysis by group."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def risk_fairness(
    df: pd.DataFrame,
    *,
    score_col: str = DEFAULT_COLS["outcome"],
    outcome_col: str = DEFAULT_COLS["treatment"],
    group_col: str = DEFAULT_COLS["gender"],
    threshold: float | None = None,
) -> pd.DataFrame:
    """Fairness metrics: FPR and FNR by demographic group.

    Binarizes the score at the given threshold (default: median) and
    computes false positive rate and false negative rate per group.

    Parameters
    ----------
    df : DataFrame
        Dataset with score, outcome, and group columns.
    score_col : str
        Column with risk score.
    outcome_col : str
        Column with binary outcome (1 = event).
    group_col : str
        Column with group labels.
    threshold : float, optional
        Classification threshold. Defaults to median of score.

    Returns
    -------
    DataFrame
        Columns: group, fpr, fnr, n.
    """
    tmp = df[[score_col, outcome_col, group_col]].dropna()
    if threshold is None:
        threshold = float(tmp[score_col].median())

    pred = (tmp[score_col] > threshold).astype(int).values
    actual = tmp[outcome_col].astype(int).values
    groups = tmp[group_col].values

    rows = []
    for g in np.unique(groups):
        mask = groups == g
        p = pred[mask]
        a = actual[mask]
        n_neg = int((a == 0).sum())
        n_pos = int((a == 1).sum())
        fp = int(((p == 1) & (a == 0)).sum())
        fn = int(((p == 0) & (a == 1)).sum())
        fpr = fp / n_neg if n_neg > 0 else np.nan
        fnr = fn / n_pos if n_pos > 0 else np.nan
        rows.append({"group": g, "fpr": fpr, "fnr": fnr, "n": int(mask.sum())})
    return pd.DataFrame(rows)


rskfr = risk_fairness


def cheatsheet() -> str:
    return "risk_fairness({}) -> Risk score fairness analysis by group."
