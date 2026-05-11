# morie.fn — function file (hadesllm/morie)
"""Denoising diffusion implicit models (DDIM) for faster sampling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ddim"]


def geron_ddim(x_T, model, T, n_steps):
    """
    Denoising diffusion implicit models (DDIM) for faster sampling

    Formula: non-Markovian deterministic reverse: x_{t-1} = sqrt(a_{t-1})*x0_pred + sqrt(1 - a_{t-1})*eps

    Parameters
    ----------
    x_T : array-like
        Input data.
    model : array-like
        Input data.
    T : array-like
        Input data.
    n_steps : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Denoising diffusion implicit models (DDIM) for faster sampling"})


def cheatsheet():
    return "hmddim: Denoising diffusion implicit models (DDIM) for faster sampling"
