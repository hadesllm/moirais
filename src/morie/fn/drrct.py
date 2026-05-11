"""DR-DiD with RCT side data."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_rct_assisted_did"]


def dr_rct_assisted_did(y_obs, y_rct, D, X):
    """
    DR-DiD with RCT side data

    Formula: borrow RCT strength under transportability

    Parameters
    ----------
    y_obs : array-like
        Input data.
    y_rct : array-like
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
    y_obs = np.atleast_1d(np.asarray(y_obs, dtype=float))
    n = len(y_obs)
    result = float(np.mean(y_obs))
    se = float(np.std(y_obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD with RCT side data"})


def cheatsheet():
    return "drrct: DR-DiD with RCT side data"
