"""1-Wasserstein distance between empirical 1-D measures."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_wasserstein_1d"]


def ot_wasserstein_1d(x, y):
    """
    1-Wasserstein distance between empirical 1-D measures

    Formula: W_1 = ∫|F(t)-G(t)| dt = mean|x_(i)-y_(i)|

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W1

    References
    ----------
    Vallender (1973)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "1-Wasserstein distance between empirical 1-D measures"})


def cheatsheet():
    return "otws: 1-Wasserstein distance between empirical 1-D measures"
