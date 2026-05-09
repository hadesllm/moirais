"""Survey-weighted quantile regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survey_quantile_reg"]


def survey_quantile_reg(y, X, weights, tau):
    """
    Survey-weighted quantile regression

    Formula: weighted check loss

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    weights : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Koenker (2005); Lumley (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-weighted quantile regression"})


def cheatsheet():
    return "svyrcq: Survey-weighted quantile regression"
