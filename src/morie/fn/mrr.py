"""Mean reciprocal rank."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mrr"]


def mrr(pred_rank, relevant):
    """
    Mean reciprocal rank

    Formula: mean(1/rank_first_correct)

    Parameters
    ----------
    pred_rank : array-like
        Input data.
    relevant : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Voorhees (1999)
    """
    pred_rank = np.atleast_1d(np.asarray(pred_rank, dtype=float))
    n = len(pred_rank)
    result = float(np.mean(pred_rank))
    se = float(np.std(pred_rank, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean reciprocal rank"})


def cheatsheet():
    return "mrr: Mean reciprocal rank"
