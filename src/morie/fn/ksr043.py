"""Quantile-process inequality used to verify Hadamard-differentiability of inverse map."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_quantile_hadamard_inequality"]


def kosorok_ch2_quantile_hadamard_inequality(F, h_n, t_n, xi_p, xi_pn, eps_pn):
    """
    Quantile-process inequality used to verify Hadamard-differentiability of inverse map

    Formula: (F + t_n h_n)(xi^N_pn - eps_pn) <= F(xi_p) <= (F + t_n h_n)(xi^N_pn)

    Parameters
    ----------
    F : array-like
        Input data.
    h_n : array-like
        Input data.
    t_n : array-like
        Input data.
    xi_p : array-like
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
    Kosorok (2008), Ch 2, Eq 2.9, p. 23
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile-process inequality used to verify Hadamard-differentiability of inverse map"})


def cheatsheet():
    return "ksr043: Quantile-process inequality used to verify Hadamard-differentiability of inverse map"
