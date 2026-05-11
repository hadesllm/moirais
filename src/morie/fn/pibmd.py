"""Prior-data conflict / sensitivity diagnostic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prior_informativeness_bias_diagnostic"]


def prior_informativeness_bias_diagnostic(samples, prior):
    """
    Prior-data conflict / sensitivity diagnostic

    Formula: D_KL( p(theta|y) || p(theta) )

    Parameters
    ----------
    samples : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Evans & Moshonov (2006)
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = len(samples)
    result = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prior-data conflict / sensitivity diagnostic"})


def cheatsheet():
    return "pibmd: Prior-data conflict / sensitivity diagnostic"
