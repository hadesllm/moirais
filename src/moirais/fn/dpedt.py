"""DP exchangeable predictive distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_exchangeable_distribution"]


def dp_exchangeable_distribution(partition, alpha):
    """
    DP exchangeable predictive distribution

    Formula: P(z_n=k|z_{1:n-1}) = n_k/(n-1+alpha)

    Parameters
    ----------
    partition : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pitman (2006)
    """
    partition = np.atleast_1d(np.asarray(partition, dtype=float))
    n = len(partition)
    result = float(np.mean(partition))
    se = float(np.std(partition, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP exchangeable predictive distribution"})


def cheatsheet():
    return "dpedt: DP exchangeable predictive distribution"
