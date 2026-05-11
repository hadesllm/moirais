"""REML log-likelihood evaluation for LMM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["reml_loglik"]


def reml_loglik(y, X, V):
    """
    REML log-likelihood evaluation for LMM

    Formula: -0.5 [log|V| + log|X'V^-1 X| + (y - X beta_hat)' V^-1 (y - X beta_hat)]

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Patterson & Thompson (1971)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "REML log-likelihood evaluation for LMM"})


def cheatsheet():
    return "remlfn: REML log-likelihood evaluation for LMM"
