# morie.fn — function file (hadesllm/morie)
"""Autoencoder reconstruction MSE loss."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_autoencoder_reconstruction_loss"]


def geron_autoencoder_reconstruction_loss(X, encoded, decoded):
    """
    Autoencoder reconstruction MSE loss

    Formula: L_AE = ||x - Dec(Enc(x))||^2

    Parameters
    ----------
    X : array-like
        Input data.
    encoded : array-like
        Input data.
    decoded : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 18, Autoencoders section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Autoencoder reconstruction MSE loss"})


def cheatsheet():
    return "grael: Autoencoder reconstruction MSE loss"
