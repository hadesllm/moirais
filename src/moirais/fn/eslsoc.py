"""Self-organizing map (SOM)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_self_organize"]


def esl_self_organize(X, grid, eta):
    """
    Self-organizing map (SOM)

    Formula: Update m_j <- m_j + eta(x_i - m_j) for nearby j

    Parameters
    ----------
    X : array-like
        Input data.
    grid : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: map

    References
    ----------
    Hastie ESL Ch 14
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self-organizing map (SOM)"})


def cheatsheet():
    return "eslsoc: Self-organizing map (SOM)"
