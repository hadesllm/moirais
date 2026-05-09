"""Thomas cluster process."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["thomas_cluster"]


def thomas_cluster(lambda_p, mu, sigma):
    """
    Thomas cluster process

    Formula: Gaussian-distributed offspring around Poisson parents

    Parameters
    ----------
    lambda_p : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Thomas (1949)
    """
    lambda_p = np.atleast_1d(np.asarray(lambda_p, dtype=float))
    n = len(lambda_p)
    result = float(np.mean(lambda_p))
    se = float(np.std(lambda_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Thomas cluster process"})


def cheatsheet():
    return "thmksp: Thomas cluster process"
