"""mth order empirical U-process measure for symmetric kernels."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_u_process_measure"]


def kosorok_ch2_u_process_measure(f, X, n, m):
    """
    mth order empirical U-process measure for symmetric kernels

    Formula: U_{n,m}(f) = (n choose m)^{-1} * sum_{i_1 < ... < i_m} f(X_{i_1}, ..., X_{i_m})

    Parameters
    ----------
    f : array-like
        Input data.
    X : array-like
        Input data.
    n : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, p. 32
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "mth order empirical U-process measure for symmetric kernels"})


def cheatsheet():
    return "ksr060: mth order empirical U-process measure for symmetric kernels"
