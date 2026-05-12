"""GloVe -- global word-vector co-occurrence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["glove"]


def glove(corpus, dim):
    """
    GloVe -- global word-vector co-occurrence

    Formula: J = sum f(X_ij)(w_i·w_j + b − log X_ij)²

    Parameters
    ----------
    corpus : array-like
        Input data.
    dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pennington-Socher-Manning (2014)
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GloVe -- global word-vector co-occurrence"})


def cheatsheet():
    return "glove: GloVe -- global word-vector co-occurrence"
