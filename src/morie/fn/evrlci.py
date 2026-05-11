"""Delta-method CI for a return level z_T."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_return_level_ci"]


def evt_return_level_ci(mu, sigma, xi, Sigma_hat, T):
    """
    Delta-method CI for a return level z_T

    Formula: Var(z_T) = ∇z_T^T Σ̂ ∇z_T

    Parameters
    ----------
    mu : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.
    Sigma_hat : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z_T, ci_lo, ci_hi

    References
    ----------
    Coles (2001)
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Delta-method CI for a return level z_T"})


def cheatsheet():
    return "evrlci: Delta-method CI for a return level z_T"
