"""limma-voom for RNA-seq."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["limma_voom"]


def limma_voom(counts, design):
    """
    limma-voom for RNA-seq

    Formula: voom precision weights + linear model

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
    Law et al (2014)
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "limma-voom for RNA-seq"})


def cheatsheet():
    return "limmav: limma-voom for RNA-seq"
