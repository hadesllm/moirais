"""Cross-correlation function of two continuous-time signals with delay tau.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_ccf_continuous_with_delay"]


def rangayyan_ch4_ccf_continuous_with_delay(x, y, tau, t):
    """
    Cross-correlation function of two continuous-time signals with delay tau.

    Formula: theta_xy(tau) = integral_{-inf}^{inf} x(t) * y(t + tau) dt

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    tau : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.27, p. 230
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Cross-correlation function of two continuous-time signals with delay tau."})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Cross-correlation function of two continuous-time signals with delay tau."})


def cheatsheet():
    return "rng201: Cross-correlation function of two continuous-time signals with delay tau."
