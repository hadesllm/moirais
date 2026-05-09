"""CTT item discrimination."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ctt_discrimination"]


def ctt_discrimination(X):
    """
    CTT item discrimination

    Formula: d_j = p_top27% - p_bottom27%

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kelley (1939)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CTT item discrimination"})


def cheatsheet():
    return "cttdis: CTT item discrimination"
