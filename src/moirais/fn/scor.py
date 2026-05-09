# moirais.fn — function file (hadesllm/moirais)
"""Inter-subscale correlation matrix."""

from __future__ import annotations

import pandas as pd

from moirais.fn._mapq_const import SUBSCALES


def subscale_correlations(
    data: pd.DataFrame,
    *,
    subscales: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Inter-subscale Pearson correlation matrix.

    Each subscale score is the mean of its items.

    Parameters
    ----------
    data : DataFrame
        Item response data.
    subscales : dict, optional
        Subscale name -> list of item column names.
        Default: MAPQ subscales (EE, EA, UA, ER).

    Returns
    -------
    DataFrame
        Correlation matrix (subscales x subscales).

    References
    ----------
    Fornell, C. & Larcker, D.F. (1981). Evaluating structural equation
        models with unobservable variables. JMR, 18(1), 39-50.
    """
    subs = subscales if subscales is not None else SUBSCALES
    scores = pd.DataFrame()
    for name, items in subs.items():
        cols = [c for c in items if c in data.columns]
        if cols:
            scores[name] = data[cols].mean(axis=1)

    return scores.corr()


def cheatsheet() -> str:
    return "subscale_correlations({}) -> Inter-subscale correlation matrix."
