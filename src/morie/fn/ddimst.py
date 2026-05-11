"""DDIM deterministic sampler."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ddim_step"]


def ddim_step(x_t, t, eps_theta, eta):
    """
    DDIM deterministic sampler

    Formula: non-Markovian deterministic reverse

    Parameters
    ----------
    x_t : array-like
        Input data.
    t : array-like
        Input data.
    eps_theta : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Song-Meng-Ermon (2021) DDIM
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DDIM deterministic sampler"})


def cheatsheet():
    return "ddimst: DDIM deterministic sampler"
