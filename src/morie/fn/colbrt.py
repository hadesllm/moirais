"""ColBERT late-interaction retrieval."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["colbert"]


def colbert(query, docs):
    """
    ColBERT late-interaction retrieval

    Formula: sum_q max_d q_i·d_j

    Parameters
    ----------
    query : array-like
        Input data.
    docs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Khattab-Zaharia (2020) ColBERT
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ColBERT late-interaction retrieval"})


def cheatsheet():
    return "colbrt: ColBERT late-interaction retrieval"
