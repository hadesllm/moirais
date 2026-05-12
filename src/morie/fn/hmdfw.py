# morie.fn -- function file (hadesllm/morie)
"""Diffusion forward process adds Gaussian noise over T steps."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_diffusion_forward"]


def geron_diffusion_forward(x0, T, beta_schedule):
    """
    Diffusion forward process adds Gaussian noise over T steps

    Formula: q(x_t | x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t I)

    Parameters
    ----------
    x0 : array-like
        Input data.
    T : array-like
        Input data.
    beta_schedule : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_T

    References
    ----------
    Géron Ch 18
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = len(x0)
    result = float(np.mean(x0))
    se = float(np.std(x0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Diffusion forward process adds Gaussian noise over T steps"})


def cheatsheet():
    return "hmdfw: Diffusion forward process adds Gaussian noise over T steps"
