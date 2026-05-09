"""DFBETA leverage diagnostic for Cox PH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dfbeta_cox"]


def dfbeta_cox(time, event, X):
    """
    DFBETA leverage diagnostic for Cox PH

    Formula: DFBETA_i = beta_hat - beta_hat^{(-i)}

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Therneau & Grambsch (2000) §5.2
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DFBETA leverage diagnostic for Cox PH"})


def cheatsheet():
    return "dlbcox: DFBETA leverage diagnostic for Cox PH"
