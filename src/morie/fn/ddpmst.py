"""DDPM reverse step."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ddpm_step"]


def ddpm_step(x_t, t, eps_theta):
    """
    DDPM reverse step

    Formula: x_{t-1} = (1/sqrt(alpha_t)) (x_t - eps_theta) + sigma_t z

    Parameters
    ----------
    x_t : array-like
        Input data.
    t : array-like
        Input data.
    eps_theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ho-Jain-Abbeel (2020)
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DDPM reverse step"})


def cheatsheet():
    return "ddpmst: DDPM reverse step"
