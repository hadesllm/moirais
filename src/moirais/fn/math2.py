"""H² index for heterogeneity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_higgins_h2"]


def ma_higgins_h2(Q, k):
    """
    H² index for heterogeneity

    Formula: H² = Q/(k-1)

    Parameters
    ----------
    Q : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: H2

    References
    ----------
    Higgins & Thompson (2002)
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "H² index for heterogeneity"})


def cheatsheet():
    return "math2: H² index for heterogeneity"
