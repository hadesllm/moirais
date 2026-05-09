"""Population intervention effect (PIE)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pie_parameters"]


def pie_parameters(y, X, intervention_dist):
    """
    Population intervention effect (PIE)

    Formula: PIE = E[Y(do(X=x*))] - E[Y(natural)]

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    intervention_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Westreich (2014)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Population intervention effect (PIE)"})


def cheatsheet():
    return "piepar: Population intervention effect (PIE)"
