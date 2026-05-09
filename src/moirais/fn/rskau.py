# moirais.fn — function file (hadesllm/moirais)
"""AUC for risk score discrimination."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS
from ._richresult import RichResult


def risk_auc(
    df: pd.DataFrame,
    *,
    score_col: str = DEFAULT_COLS["outcome"],
    outcome_col: str = DEFAULT_COLS["treatment"],
) -> dict:
    """Area under the ROC curve for a risk score.

    Uses the Mann-Whitney U statistic formulation:
    AUC = U / (n1 * n0).

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
        auc, ci_lower, ci_upper (DeLong approximate 95% CI),
        n_pos, n_neg.
    """
    tmp = df[[score_col, outcome_col]].dropna()
    scores = tmp[score_col].values.astype(float)
    labels = tmp[outcome_col].values.astype(int)

    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n1 = len(pos)
    n0 = len(neg)

    if n1 == 0 or n0 == 0:
        return RichResult(payload={"auc": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_pos": n1, "n_neg": n0})

    # Mann-Whitney U
    u = 0.0
    for p in pos:
        u += np.sum(scores[labels == 0] < p) + 0.5 * np.sum(scores[labels == 0] == p)
    auc = u / (n1 * n0)

    # Hanley-McNeil SE approximation
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
    ci_lower = max(0.0, auc - 1.96 * se)
    ci_upper = min(1.0, auc + 1.96 * se)

    return {
        "auc": float(auc),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_pos": n1,
        "n_neg": n0,
    }


rskau = risk_auc


def cheatsheet() -> str:
    return "risk_auc({}) -> AUC for risk score discrimination."
