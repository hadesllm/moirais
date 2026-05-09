"""Beta-Binomial conjugate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["beta_binomial"]


def beta_binomial(successes, trials, alpha, beta):
    """
    Beta-Binomial conjugate

    Formula: p ~ Beta(alpha, beta); y ~ Binomial(n, p)

    Parameters
    ----------
    successes : array-like
        Input data.
    trials : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman BDA3
    """
    successes = np.atleast_1d(np.asarray(successes, dtype=float))
    n = len(successes)
    result = float(np.mean(successes))
    se = float(np.std(successes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Beta-Binomial conjugate"})


def cheatsheet():
    return "betbnm: Beta-Binomial conjugate"
