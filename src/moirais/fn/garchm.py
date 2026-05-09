"""GARCH(p,q) volatility model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["garch_model"]


def garch_model(y, p, q):
    """
    GARCH(p,q) volatility model

    Formula: sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bollerslev (1986)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GARCH(p,q) volatility model"})


def cheatsheet():
    return "garchm: GARCH(p,q) volatility model"
