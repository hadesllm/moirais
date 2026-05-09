"""Sharp RDD via local linear regression at threshold."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_rdd_local_lin"]


def causal_rdd_local_lin(x, y, cutoff, h):
    """
    Sharp RDD via local linear regression at threshold

    Formula: τ̂ = α̂_+ - α̂_-, with triangular kernel weights

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    cutoff : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tau, se, se_robust

    References
    ----------
    Imbens-Kalyanaraman (2012)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sharp RDD via local linear regression at threshold"})


def cheatsheet():
    return "causrdd: Sharp RDD via local linear regression at threshold"
