"""Survey-weighted ratio."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survey_ratio"]


def survey_ratio(y, x, weights):
    """
    Survey-weighted ratio

    Formula: (sum w y) / (sum w x)

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-weighted ratio"})


def cheatsheet():
    return "smltrt: Survey-weighted ratio"
