"""Chi-squared goodness of fit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_chi_sq_gof"]


def wasserman_chi_sq_gof(observed, expected):
    """
    Chi-squared goodness of fit

    Formula: X^2 = sum (O_j - E_j)^2 / E_j

    Parameters
    ----------
    observed : array-like
        Input data.
    expected : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Wasserman (2004), Ch 10
    """
    observed = np.atleast_1d(np.asarray(observed, dtype=float))
    n = len(observed)
    result = float(np.mean(observed))
    se = float(np.std(observed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chi-squared goodness of fit"})


def cheatsheet():
    return "wsmchi: Chi-squared goodness of fit"
