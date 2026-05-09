"""Spearman-Brown split-half reliability."""

from __future__ import annotations

import numpy as np
import pandas as pd


def splhf(
    data: pd.DataFrame | np.ndarray,
    method: str = "first_last",
) -> float:
    """Spearman-Brown split-half reliability.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    method : str
        Split method: ``'first_last'`` (first half vs second half) or
        ``'odd_even'`` (odd-indexed vs even-indexed items).

    Returns
    -------
    float
        Spearman-Brown corrected split-half reliability coefficient.
    """
    X = np.asarray(data, dtype=np.float64)
    k = X.shape[1]

    if method == "odd_even":
        h1 = X[:, 0::2].sum(axis=1)
        h2 = X[:, 1::2].sum(axis=1)
    else:
        mid = k // 2
        h1 = X[:, :mid].sum(axis=1)
        h2 = X[:, mid:].sum(axis=1)

    r = float(np.corrcoef(h1, h2)[0, 1])
    return 2 * r / (1 + r)


def cheatsheet() -> str:
    return "splhf({}) -> Spearman-Brown split-half reliability."
