"""Warner classic randomized response (sensitive yes/no)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["randomized_response"]


def randomized_response(y, truth, p):
    """
    Warner classic randomized response (sensitive yes/no)

    Formula: observed = p * truth + (1-p) * Bernoulli(0.5)

    Parameters
    ----------
    y : array-like
        Input data.
    truth : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Warner (1965)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Warner classic randomized response (sensitive yes/no)"})


def cheatsheet():
    return "randres: Warner classic randomized response (sensitive yes/no)"
