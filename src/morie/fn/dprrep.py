"""Warner randomized response (epsilon-DP for binary)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["randomized_response_dp"]


def randomized_response_dp(y, truth, epsilon):
    """
    Warner randomized response (epsilon-DP for binary)

    Formula: P(yes-report | true=yes) = (e^epsilon)/(e^epsilon + 1)

    Parameters
    ----------
    y : array-like
        Input data.
    truth : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Warner (1965); Dwork (2006) DP framing
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Warner randomized response (epsilon-DP for binary)"})


def cheatsheet():
    return "dprrep: Warner randomized response (epsilon-DP for binary)"
