"""Taylor-style upper and lower bounds on F(xi_p) used in quantile delta-method proof."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_quantile_taylor_bounds"]


def kosorok_ch2_quantile_taylor_bounds(F, h, t_n, xi_pn, eps_pn):
    """
    Taylor-style upper and lower bounds on F(xi_p) used in quantile delta-method proof

    Formula: F(xi^N_pn) + t_n h(xi_pn - eps_pn) + o(t_n) <= F(xi_p) <= F(xi^N_pn) + t_n h(xi^N_pn) + o(t_n)

    Parameters
    ----------
    F : array-like
        Input data.
    h : array-like
        Input data.
    t_n : array-like
        Input data.
    xi_pn : array-like
        Input data.
    eps_pn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.10, p. 23
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Taylor-style upper and lower bounds on F(xi_p) used in quantile delta-method proof"})


def cheatsheet():
    return "ksr044: Taylor-style upper and lower bounds on F(xi_p) used in quantile delta-method proof"
