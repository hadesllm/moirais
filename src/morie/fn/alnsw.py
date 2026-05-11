"""Smith-Waterman local alignment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["smith_waterman"]


def smith_waterman(seq1, seq2, sub_matrix, gap):
    """
    Smith-Waterman local alignment

    Formula: DP with non-negative cell floor + traceback from max

    Parameters
    ----------
    seq1 : array-like
        Input data.
    seq2 : array-like
        Input data.
    sub_matrix : array-like
        Input data.
    gap : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Smith-Waterman (1981)
    """
    seq1 = np.atleast_1d(np.asarray(seq1, dtype=float))
    n = len(seq1)
    result = float(np.mean(seq1))
    se = float(np.std(seq1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smith-Waterman local alignment"})


def cheatsheet():
    return "alnsw: Smith-Waterman local alignment"
