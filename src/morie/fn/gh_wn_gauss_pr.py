# morie.fn — function file (hadesllm/morie)
"""White noise model with conjugate Gaussian prior: exact posterior Gaussian."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_white_noise_gauss_prior"]


def ghosal_white_noise_gauss_prior(x):
    """
    White noise model with conjugate Gaussian prior: exact posterior Gaussian

    Formula: dY = theta dt + dW/sqrt(n), theta ~ GP(0,C) => theta|Y ~ N(posterior mean, posterior covariance)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 9 §9.5.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "White noise model with conjugate Gaussian prior: exact posterior Gaussian"})


def cheatsheet():
    return "gh_wn_gauss_pr: White noise model with conjugate Gaussian prior: exact posterior Gaussian"
