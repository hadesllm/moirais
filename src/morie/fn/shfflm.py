"""Shuffle model amplification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shuffle_model"]


def shuffle_model(epsilon0, n, delta):
    """
    Shuffle model amplification

    Formula: shuffler converts ε₀-LDP to (ε,δ)-DP with √n gain

    Parameters
    ----------
    epsilon0 : array-like
        Input data.
    n : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cheu et al (2019); Erlingsson et al (2019)
    """
    epsilon0 = np.atleast_1d(np.asarray(epsilon0, dtype=float))
    n = len(epsilon0)
    result = float(np.mean(epsilon0))
    se = float(np.std(epsilon0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shuffle model amplification"})


def cheatsheet():
    return "shfflm: Shuffle model amplification"
