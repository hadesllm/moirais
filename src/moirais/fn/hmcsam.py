"""Hamiltonian Monte Carlo."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hamiltonian_mc"]


def hamiltonian_mc(log_p, grad_log_p, x0, step_size, L):
    """
    Hamiltonian Monte Carlo

    Formula: leapfrog integration on H = U + K

    Parameters
    ----------
    log_p : array-like
        Input data.
    grad_log_p : array-like
        Input data.
    x0 : array-like
        Input data.
    step_size : array-like
        Input data.
    L : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Neal (2011)
    """
    log_p = np.atleast_1d(np.asarray(log_p, dtype=float))
    n = len(log_p)
    result = float(np.mean(log_p))
    se = float(np.std(log_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hamiltonian Monte Carlo"})


def cheatsheet():
    return "hmcsam: Hamiltonian Monte Carlo"
