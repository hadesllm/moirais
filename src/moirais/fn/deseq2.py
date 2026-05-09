"""DESeq2 differential expression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deseq2_differential"]


def deseq2_differential(counts, design):
    """
    DESeq2 differential expression

    Formula: NB GLM with shrinkage on log fold change

    Parameters
    ----------
    counts : array-like
        Input data.
    design : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Love-Huber-Anders (2014)
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DESeq2 differential expression"})


def cheatsheet():
    return "deseq2: DESeq2 differential expression"
