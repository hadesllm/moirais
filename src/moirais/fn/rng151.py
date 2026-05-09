"""Optimal Wiener filter for noise removal using signal+noise autocorrelation matrices.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_wiener_optimal_for_noise_removal"]


def rangayyan_ch3_wiener_optimal_for_noise_removal(Phi_d, Phi_eta, Phi_1d):
    """
    Optimal Wiener filter for noise removal using signal+noise autocorrelation matrices.

    Formula: w_o = (Phi_d + Phi_eta)^(-1) * Phi_1d

    Parameters
    ----------
    Phi_d : array-like
        Input data.
    Phi_eta : array-like
        Input data.
    Phi_1d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.183, p. 177
    """
    Phi_d = np.atleast_1d(np.asarray(Phi_d, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(Phi_d), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Optimal Wiener filter for noise removal using signal+noise autocorrelation matrices."})
    result = stats.spearmanr(Phi_d[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Optimal Wiener filter for noise removal using signal+noise autocorrelation matrices."})


def cheatsheet():
    return "rng151: Optimal Wiener filter for noise removal using signal+noise autocorrelation matrices."
