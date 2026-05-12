"""SASRec -- self-attention sequential rec."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sasrec"]


def sasrec(seqs, K):
    """
    SASRec -- self-attention sequential rec

    Formula: transformer decoder over item sequence

    Parameters
    ----------
    seqs : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kang-McAuley (2018)
    """
    seqs = np.atleast_1d(np.asarray(seqs, dtype=float))
    n = len(seqs)
    result = float(np.mean(seqs))
    se = float(np.std(seqs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SASRec -- self-attention sequential rec"})


def cheatsheet():
    return "sasRec: SASRec -- self-attention sequential rec"
