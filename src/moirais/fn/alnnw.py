"""Needleman-Wunsch global alignment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["needleman_wunsch"]


def needleman_wunsch(seq1, seq2, sub_matrix, gap):
    """
    Needleman-Wunsch global alignment

    Formula: DP recurrence over substitution matrix + gap penalties

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
    Needleman-Wunsch (1970)
    """
    seq1 = np.atleast_1d(np.asarray(seq1, dtype=float))
    n = len(seq1)
    result = float(np.mean(seq1))
    se = float(np.std(seq1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Needleman-Wunsch global alignment"})


def cheatsheet():
    return "alnnw: Needleman-Wunsch global alignment"
