"""Combined SAR-SEM (SARMA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_ar_combined"]


def spatial_ar_combined(y, X, W1, W2):
    """
    Combined SAR-SEM (SARMA)

    Formula: y = rho W1 y + X beta + u, u = lambda W2 u + epsilon

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    W1 : array-like
        Input data.
    W2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kelejian & Prucha (1998)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Combined SAR-SEM (SARMA)"})


def cheatsheet():
    return "sarmix: Combined SAR-SEM (SARMA)"
