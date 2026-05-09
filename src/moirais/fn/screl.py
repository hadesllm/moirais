# moirais.fn — function file (hadesllm/moirais)
"""Score-level reliability summary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn.crba import crba


def score_reliability(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> dict:
    """Score-level reliability summary: alpha, SEM, MDC.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    items : list[str] or None
        Subset of columns. If None, use all numeric columns.

    Returns
    -------
    dict
        Keys: 'alpha', 'k', 'n', 'sd_total', 'sem' (standard error of
        measurement), 'mdc_90' (minimal detectable change at 90% CI),
        'mdc_95'.

    References
    ----------
    Weir, J. P. (2005). Quantifying test-retest reliability using the
    intraclass correlation coefficient and the SEM. Journal of Strength
    and Conditioning Research, 19(1), 231-240.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])
    subset = data[items] if items is not None else data.select_dtypes(include=[np.number])

    res = crba(subset)
    alpha = res.raw
    total = subset.sum(axis=1)
    sd_total = float(np.std(total, ddof=1))

    if np.isnan(alpha) or alpha < 0:
        sem = np.nan
    else:
        sem = sd_total * np.sqrt(1 - alpha)

    mdc_90 = sem * 1.645 * np.sqrt(2) if not np.isnan(sem) else np.nan
    mdc_95 = sem * 1.96 * np.sqrt(2) if not np.isnan(sem) else np.nan

    return {
        "alpha": float(alpha),
        "k": res.k,
        "n": res.n,
        "sd_total": sd_total,
        "sem": float(sem) if not np.isnan(sem) else np.nan,
        "mdc_90": float(mdc_90) if not np.isnan(mdc_90) else np.nan,
        "mdc_95": float(mdc_95) if not np.isnan(mdc_95) else np.nan,
    }


def cheatsheet() -> str:
    return "score_reliability({}) -> Score-level reliability summary."
