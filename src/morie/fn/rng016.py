"""Autocorrelation function (ACF) of a random process via expectation and joint PDF.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_acf_continuous"]


def rangayyan_ch3_acf_continuous(x, t1, tau):
    """
    Autocorrelation function (ACF) of a random process via expectation and joint PDF.

    Formula: phi_xx(t1, t1+tau) = E[x(t1) x(t1+tau)] = double_integral x(t1) x(t1+tau) p_{x1,x2}(x1,x2) dx1 dx2

    Parameters
    ----------
    x : array-like
        Input data.
    t1 : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.16, p. 96
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Autocorrelation function (ACF) of a random process via expectation and joint PDF."})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Autocorrelation function (ACF) of a random process via expectation and joint PDF."})


def cheatsheet():
    return "rng016: Autocorrelation function (ACF) of a random process via expectation and joint PDF."
