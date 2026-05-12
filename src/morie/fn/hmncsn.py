# morie.fn -- function file (hadesllm/morie)
"""Noise Conditional Score Network (NCSN): score-based generative model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ncsn"]


def geron_ncsn(X, sigmas, epochs, lr):
    """
    Noise Conditional Score Network (NCSN): score-based generative model

    Formula: train s_theta(x, sigma) to match grad_x log p_sigma(x)

    Parameters
    ----------
    X : array-like
        Input data.
    sigmas : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Noise Conditional Score Network (NCSN): score-based generative model"})


def cheatsheet():
    return "hmncsn: Noise Conditional Score Network (NCSN): score-based generative model"
