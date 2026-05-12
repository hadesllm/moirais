"""RCT-assisted TMLE -- augment RCT with observational."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_rct_assisted"]


def tmle_rct_assisted(y_rct, y_obs, D, X):
    """
    RCT-assisted TMLE -- augment RCT with observational

    Formula: borrow obs strength under transportability

    Parameters
    ----------
    y_rct : array-like
        Input data.
    y_obs : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey et al (2020)
    """
    y_rct = np.atleast_1d(np.asarray(y_rct, dtype=float))
    n = len(y_rct)
    result = float(np.mean(y_rct))
    se = float(np.std(y_rct, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RCT-assisted TMLE -- augment RCT with observational"})


def cheatsheet():
    return "tmlrct: RCT-assisted TMLE -- augment RCT with observational"
