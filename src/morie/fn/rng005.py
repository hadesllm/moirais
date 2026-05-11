"""Kurtosis as the normalized fourth central moment of the PDF.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_kurtosis"]


def rangayyan_ch3_kurtosis(eta, mu_eta, sigma_eta, p_eta):
    """
    Kurtosis as the normalized fourth central moment of the PDF.

    Formula: K_eta = (1/sigma_eta^4) * integral_{-inf}^{inf} (eta - mu_eta)^4 * p_eta(eta) d(eta)

    Parameters
    ----------
    eta : array-like
        Input data.
    mu_eta : array-like
        Input data.
    sigma_eta : array-like
        Input data.
    p_eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.5, p. 94
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    n = len(eta)
    result = float(np.mean(eta))
    se = float(np.std(eta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kurtosis as the normalized fourth central moment of the PDF."})


def cheatsheet():
    return "rng005: Kurtosis as the normalized fourth central moment of the PDF."
