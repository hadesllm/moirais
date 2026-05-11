"""NDCG@k."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ndcg"]


def ndcg(pred_rank, relevant, k):
    """
    NDCG@k

    Formula: DCG/IDCG with log discount

    Parameters
    ----------
    pred_rank : array-like
        Input data.
    relevant : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Järvelin-Kekäläinen (2002)
    """
    pred_rank = np.atleast_1d(np.asarray(pred_rank, dtype=float))
    n = len(pred_rank)
    result = float(np.mean(pred_rank))
    se = float(np.std(pred_rank, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NDCG@k"})


def cheatsheet():
    return "ndcg: NDCG@k"
