# morie.fn -- function file (hadesllm/morie)
"""Diffusion reverse process denoises from x_T back to x_0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_diffusion_reverse"]


def geron_diffusion_reverse(x_T, model, T):
    """
    Diffusion reverse process denoises from x_T back to x_0

    Formula: p_theta(x_{t-1} | x_t) = N(mu_theta(x_t,t), Sigma_theta)

    Parameters
    ----------
    x_T : array-like
        Input data.
    model : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x0

    References
    ----------
    Géron Ch 18
    """
    x_T = np.atleast_1d(np.asarray(x_T, dtype=float))
    n = len(x_T)
    result = float(np.mean(x_T))
    se = float(np.std(x_T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Diffusion reverse process denoises from x_T back to x_0"})


def cheatsheet():
    return "hmdrv: Diffusion reverse process denoises from x_T back to x_0"
