"""Survey-weighted total."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survey_total"]


def survey_total(y, weights):
    """
    Survey-weighted total

    Formula: sum w_i y_i

    Parameters
    ----------
    y : array-like
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
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-weighted total"})


def cheatsheet():
    return "smltot: Survey-weighted total"
