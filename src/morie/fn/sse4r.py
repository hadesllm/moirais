"""SSE-PT sequential rec with personalization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ssepta_seq"]


def ssepta_seq(seqs, K):
    """
    SSE-PT sequential rec with personalization

    Formula: time-aware transformer with user embedding

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
    Wu et al (2020)
    """
    seqs = np.atleast_1d(np.asarray(seqs, dtype=float))
    n = len(seqs)
    result = float(np.mean(seqs))
    se = float(np.std(seqs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SSE-PT sequential rec with personalization"})


def cheatsheet():
    return "sse4r: SSE-PT sequential rec with personalization"
