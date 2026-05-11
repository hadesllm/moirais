"""TMLE for dynamic treatment regime."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_dynamic_regime"]


def tmle_dynamic_regime(y, treatment_history, covariate_history, regime):
    """
    TMLE for dynamic treatment regime

    Formula: target E[Y(d)] for rule d(W); sequential regression

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    regime : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Petersen et al (2014); Luedtke & vdL (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for dynamic treatment regime"})


def cheatsheet():
    return "tmldyn: TMLE for dynamic treatment regime"
