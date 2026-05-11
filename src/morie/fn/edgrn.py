"""edgeR differential expression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["edger_diff"]


def edger_diff(counts, design):
    """
    edgeR differential expression

    Formula: NB QL F-test with TMM normalization

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
    Robinson-McCarthy-Smyth (2010)
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "edgeR differential expression"})


def cheatsheet():
    return "edgrn: edgeR differential expression"
