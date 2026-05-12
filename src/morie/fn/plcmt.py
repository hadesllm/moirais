# morie.fn — function file (hadesllm/morie)
"""Ranks, block frequencies, and placements (Gibbons Ch 2.11.3).

Returns the rank-vector of Y observations among the combined
sample plus their "placement" indices (count of X's less than
each Y_j).  Placements P_j = (# of X_i < Y_j) and the
Mann-Whitney statistic U_y = sum(P_j).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["rank_placements"]


def rank_placements(x, y):
    """Compute placements of Y among X order statistics.

    Parameters
    ----------
    x, y : array-like
        Independent univariate samples.

    Returns
    -------
    RichResult with payload:
        placements   : (n,) vector, P_j = #{i : X_i < Y_j} for each Y_j
        ranks_y      : ranks of Y in pooled (X, Y) sample
        U_y          : Mann-Whitney U = sum(placements)
        E_U          : expected U under H0 = m*n/2
        Var_U        : variance under H0 = m*n*(m+n+1)/12
        m, n         : sample sizes
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = int(x.size)
    n = int(y.size)
    if m < 1 or n < 1:
        return RichResult(payload={
            "placements": np.array([]),
            "ranks_y": np.array([]),
            "U_y": np.nan,
            "E_U": np.nan,
            "Var_U": np.nan,
            "m": m, "n": n,
            "method": "Rank placements",
        })
    xs = np.sort(x)
    # Placement of each Y_j among X's = number of X_i strictly less than Y_j
    placements = np.searchsorted(xs, y, side="left").astype(int)
    # Pooled-sample ranks for Y
    pooled = np.concatenate([x, y])
    all_ranks = stats.rankdata(pooled)
    ranks_y = all_ranks[m:]
    U_y = float(placements.sum())
    E_U = m * n / 2.0
    Var_U = m * n * (m + n + 1.0) / 12.0
    return RichResult(payload={
        "placements": placements,
        "ranks_y": ranks_y,
        "U_y": U_y,
        "E_U": E_U,
        "Var_U": Var_U,
        "m": m,
        "n": n,
        "method": "Rank placements",
    })


def cheatsheet():
    return "plcmt: Rank placements of Y among X order statistics"


# CANONICAL TEST
# >>> rank_placements([1, 3, 5], [2, 4, 6])
# placements = [1, 2, 3]; U_y = 6; E[U] = 4.5; Var[U] = 5.25
