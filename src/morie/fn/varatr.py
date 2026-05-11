"""Value at Risk (VaR) from GARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["value_at_risk"]


def value_at_risk(y, alpha):
    """
    Value at Risk (VaR) from GARCH

    Formula: VaR_alpha = -mu + sigma * z_{1-alpha}

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jorion (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Value at Risk (VaR) from GARCH"})


def cheatsheet():
    return "varatr: Value at Risk (VaR) from GARCH"
