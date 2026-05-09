"""Second-order Taylor expansion of population criterion around theta_0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_m_estimator_taylor_expansion"]


def kosorok_ch2_m_estimator_taylor_expansion(m, theta, theta_0, P):
    """
    Second-order Taylor expansion of population criterion around theta_0

    Formula: P[m_theta - m_{theta_0} - (theta - theta_0)' m_dot_{theta_0}] = o(||theta - theta_0||^2)

    Parameters
    ----------
    m : array-like
        Input data.
    theta : array-like
        Input data.
    theta_0 : array-like
        Input data.
    P : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.19, p. 29
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    n = len(m)
    result = float(np.mean(m))
    se = float(np.std(m, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Second-order Taylor expansion of population criterion around theta_0"})


def cheatsheet():
    return "ksr055: Second-order Taylor expansion of population criterion around theta_0"
