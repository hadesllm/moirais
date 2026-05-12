# morie.fn — function file (hadesllm/morie)
"""Rank-order statistics for paired samples (Gibbons Ch 5.5).

For a single sample of paired differences D_i = X_i - mu0, return
the signed ranks R_i^+ = sign(D_i) * rank(|D_i|).  Used as the
input layer for Wilcoxon signed-rank.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["rank_order_statistics"]


def rank_order_statistics(x, mu0: float = 0.0):
    """Signed ranks of |D_i| for paired-sample inputs.

    Parameters
    ----------
    x : array-like
        Sample of differences (or values, with ``mu0`` subtracted).
    mu0 : float
        Hypothesised median (default 0).

    Returns
    -------
    RichResult with payload:
        signed_ranks : (n,) array of signed ranks
        abs_ranks    : ranks of |D_i|
        W_plus       : sum of positive signed ranks (Wilcoxon T+)
        W_minus      : sum of |negative signed ranks|
        n_nonzero    : count of non-zero differences
        n            : input size
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 2:
        return RichResult(payload={
            "signed_ranks": np.array([]),
            "abs_ranks": np.array([]),
            "W_plus": np.nan, "W_minus": np.nan,
            "n_nonzero": 0, "n": n,
            "method": "Rank-order signed ranks",
        })
    d = x - float(mu0)
    nz = d != 0
    if not nz.any():
        return RichResult(payload={
            "signed_ranks": np.zeros(n),
            "abs_ranks": np.zeros(n),
            "W_plus": 0.0, "W_minus": 0.0,
            "n_nonzero": 0, "n": n,
            "method": "Rank-order signed ranks",
        })
    abs_ranks_nz = stats.rankdata(np.abs(d[nz]))
    signed_ranks = np.zeros(n)
    signed_ranks[nz] = np.sign(d[nz]) * abs_ranks_nz
    abs_ranks = np.zeros(n)
    abs_ranks[nz] = abs_ranks_nz
    W_plus = float(signed_ranks[signed_ranks > 0].sum())
    W_minus = float(-signed_ranks[signed_ranks < 0].sum())
    return RichResult(payload={
        "signed_ranks": signed_ranks,
        "abs_ranks": abs_ranks,
        "W_plus": W_plus,
        "W_minus": W_minus,
        "n_nonzero": int(nz.sum()),
        "n": n,
        "method": "Rank-order signed ranks",
    })


def cheatsheet():
    return "rnkor: Rank-order signed ranks for paired samples"


# CANONICAL TEST
# >>> rank_order_statistics([-2, -1, 1, 2, 3])
# |D|=[2,1,1,2,3]; ranks=[3.5,1.5,1.5,3.5,5]; W_plus=10, W_minus=5
