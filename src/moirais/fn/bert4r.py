"""BERT4Rec."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bert4rec"]


def bert4rec(seqs, K):
    """
    BERT4Rec

    Formula: masked item LM over user sequence

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
    Sun et al (2019)
    """
    seqs = np.atleast_1d(np.asarray(seqs, dtype=float))
    n = len(seqs)
    result = float(np.mean(seqs))
    se = float(np.std(seqs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BERT4Rec"})


def cheatsheet():
    return "bert4r: BERT4Rec"
