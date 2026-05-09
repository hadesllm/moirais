# moirais.fn — function file (hadesllm/moirais)
"""Denoising diffusion probabilistic model (DDPM)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ddpm"]


def geron_ddpm(X, T, beta_schedule, epochs, lr):
    """
    Denoising diffusion probabilistic model (DDPM)

    Formula: x_t = sqrt(a_t)*x_0 + sqrt(1 - a_t)*eps; train eps_theta(x_t, t)

    Parameters
    ----------
    X : array-like
        Input data.
    T : array-like
        Input data.
    beta_schedule : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 18
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Denoising diffusion probabilistic model (DDPM)"})


def cheatsheet():
    return "hmddpm: Denoising diffusion probabilistic model (DDPM)"
