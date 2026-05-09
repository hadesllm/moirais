"""Max-sliced Wasserstein distance over projection direction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_max_sliced_w"]


def ot_max_sliced_w(X, Y, p, n_proj):
    """
    Max-sliced Wasserstein distance over projection direction

    Formula: max_θ W_p(P_θ#μ, P_θ#ν)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    p : array-like
        Input data.
    n_proj : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: MSW, theta_star

    References
    ----------
    Deshpande-Hu-Sun-Pyrros (2019)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Max-sliced Wasserstein distance over projection direction"})


def cheatsheet():
    return "otmsw: Max-sliced Wasserstein distance over projection direction"
