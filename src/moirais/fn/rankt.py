# moirais.fn — function file (hadesllm/moirais)
"""Rank transformation for nonparametric analysis."""

import numpy as np

from ._containers import DescriptiveResult


def rank_transform(x, method="average"):
    """
    Compute rank transformation of data.

    Converts raw scores to ranks, handling ties with the specified method.
    Useful for rank-based nonparametric methods and robust statistics.

    :param x: (n,) numeric data.
    :param method: Tie-breaking: 'average', 'min', 'max', 'dense', 'ordinal'.
    :return: DescriptiveResult with ranks and tie information.

    References
    ----------
    Conover WJ & Iman RL (1981). Rank Transformations as a Bridge
    Between Parametric and Nonparametric Statistics. Am Stat 35(3):124-129.
    """
    from scipy.stats import rankdata

    arr = np.asarray(x, dtype=np.float64).ravel()
    ranks = rankdata(arr, method=method)

    unique, counts = np.unique(arr, return_counts=True)
    n_ties = int(np.sum(counts > 1))
    tie_fraction = float(n_ties / len(unique)) if len(unique) > 0 else 0.0

    return DescriptiveResult(
        name="rank_transform",
        value=float(np.mean(ranks)),
        extra={
            "ranks": ranks.tolist(),
            "method": method,
            "n": len(arr),
            "n_ties": n_ties,
            "tie_fraction": float(tie_fraction),
            "n_unique": int(len(unique)),
        },
    )


def cheatsheet() -> str:
    return "rank_transform({}) -> Rank transformation for nonparametric analysis."
