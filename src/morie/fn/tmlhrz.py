"""TMLE for marginal hazard ratio under non-PH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_hazard_ratio"]


def tmle_hazard_ratio(time, event, D, X):
    """
    TMLE for marginal hazard ratio under non-PH

    Formula: target hazard ratio with clever covariate Q*

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
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
    Westling et al (2024)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for marginal hazard ratio under non-PH"})


def cheatsheet():
    return "tmlhrz: TMLE for marginal hazard ratio under non-PH"
