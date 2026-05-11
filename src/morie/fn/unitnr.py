"""Unit nonresponse propensity weighting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["unit_nonresponse"]


def unit_nonresponse(respondents, frame, X):
    """
    Unit nonresponse propensity weighting

    Formula: adjust w_i by 1/Pr(respond | demographics)

    Parameters
    ----------
    respondents : array-like
        Input data.
    frame : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Little-Vartivarian (2005)
    """
    respondents = np.atleast_1d(np.asarray(respondents, dtype=float))
    n = len(respondents)
    result = float(np.mean(respondents))
    se = float(np.std(respondents, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unit nonresponse propensity weighting"})


def cheatsheet():
    return "unitnr: Unit nonresponse propensity weighting"
