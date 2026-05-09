"""Variance of a random process defined as the second central moment of its PDF.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_variance_continuous"]


def rangayyan_ch3_variance_continuous(eta, mu_eta, p_eta):
    """
    Variance of a random process defined as the second central moment of its PDF.

    Formula: sigma_eta^2 = E[(eta - mu_eta)^2] = integral_{-inf}^{inf} (eta - mu_eta)^2 * p_eta(eta) d(eta)

    Parameters
    ----------
    eta : array-like
        Input data.
    mu_eta : array-like
        Input data.
    p_eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.3, p. 94
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    n = len(eta)
    result = float(np.mean(eta))
    se = float(np.std(eta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance of a random process defined as the second central moment of its PDF."})


def cheatsheet():
    return "rng003: Variance of a random process defined as the second central moment of its PDF."
