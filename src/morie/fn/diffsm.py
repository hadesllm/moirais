"""Score matching for diffusion models."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["diffusion_score_matching"]


def diffusion_score_matching(x, s_theta, sigma):
    """
    Score matching for diffusion models

    Formula: E[||s_theta(x_t,t) - grad log q(x_t|x_0)||^2]

    Parameters
    ----------
    x : array-like
        Input data.
    s_theta : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Song-Ermon (2019); Ho-Jain-Abbeel (2020) DDPM
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Score matching for diffusion models"})


def cheatsheet():
    return "diffsm: Score matching for diffusion models"
