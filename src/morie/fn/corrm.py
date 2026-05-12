# morie.fn -- function file (hadesllm/morie)
"""Full correlation matrix with p-values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def correlation_matrix(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
) -> DescriptiveResult:
    """Correlation matrix with two-tailed p-values.

    Parameters
    ----------
    df : DataFrame
        Numeric columns only.
    method : str
        'pearson', 'spearman', or 'kendall'.

    Returns
    -------
    DescriptiveResult
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(np.asarray(df, dtype=float))

    numeric = df.select_dtypes(include=[np.number])
    cols = list(numeric.columns)
    k = len(cols)
    if k < 2:
        raise ValueError("Need >= 2 numeric columns.")

    rmat = np.ones((k, k))
    pmat = np.zeros((k, k))

    fn = {"pearson": sp_stats.pearsonr, "spearman": sp_stats.spearmanr, "kendall": sp_stats.kendalltau}.get(method)
    if fn is None:
        raise ValueError(f"Unknown method: {method}")

    for i in range(k):
        for j in range(i + 1, k):
            a = numeric.iloc[:, i].dropna()
            b = numeric.iloc[:, j].dropna()
            idx = a.index.intersection(b.index)
            r, p = fn(a.loc[idx], b.loc[idx])
            rmat[i, j] = rmat[j, i] = r
            pmat[i, j] = pmat[j, i] = p

    r_df = pd.DataFrame(rmat, index=cols, columns=cols)
    p_df = pd.DataFrame(pmat, index=cols, columns=cols)

    return DescriptiveResult(
        name="correlation_matrix",
        value=r_df,
        extra={"p_values": p_df, "method": method, "n_vars": k},
    )


corrm = correlation_matrix


def cheatsheet() -> str:
    return "correlation_matrix({}) -> Full correlation matrix with p-values."
