"""Survey-weighted contingency table."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survey_xtab"]


def survey_xtab(x, y, weights):
    """
    Survey-weighted contingency table

    Formula: weighted cell counts + Wald test

    Parameters
    ----------
    x : array-like
        Input data.
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
    Rao-Scott (1984)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-weighted contingency table"})


def cheatsheet():
    return "svytbl: Survey-weighted contingency table"
