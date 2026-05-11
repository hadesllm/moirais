"""Posterior mean E[theta|x]."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_posterior_mean"]


def wasserman_posterior_mean(posterior):
    """
    Posterior mean E[theta|x]

    Formula: theta_bayes = int theta p(theta|x) d theta

    Parameters
    ----------
    posterior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 11
    """
    posterior = np.atleast_1d(np.asarray(posterior, dtype=float))
    n = len(posterior)
    result = float(np.mean(posterior))
    se = float(np.std(posterior, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior mean E[theta|x]"})


def cheatsheet():
    return "wsmpst1: Posterior mean E[theta|x]"
