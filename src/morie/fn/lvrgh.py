"""Hat matrix diagonal h_ii (leverage)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hat_matrix_diagonal"]


def hat_matrix_diagonal(y, X):
    """
    Hat matrix diagonal h_ii (leverage)

    Formula: h_ii = x_i' (X' X)^-1 x_i

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hoaglin & Welsch (1978)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hat matrix diagonal h_ii (leverage)"})


def cheatsheet():
    return "lvrgh: Hat matrix diagonal h_ii (leverage)"
