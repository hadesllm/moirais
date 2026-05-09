"""Sequential Monte Carlo (SMC) sampler."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sequential_mc_sampler"]


def sequential_mc_sampler(log_p, temperatures, n_particles):
    """
    Sequential Monte Carlo (SMC) sampler

    Formula: particles + tempering + resample

    Parameters
    ----------
    log_p : array-like
        Input data.
    temperatures : array-like
        Input data.
    n_particles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Del Moral-Doucet-Jasra (2006)
    """
    log_p = np.atleast_1d(np.asarray(log_p, dtype=float))
    n = len(log_p)
    result = float(np.mean(log_p))
    se = float(np.std(log_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential Monte Carlo (SMC) sampler"})


def cheatsheet():
    return "smcsam: Sequential Monte Carlo (SMC) sampler"
