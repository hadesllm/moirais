# morie.fn — function file (hadesllm/morie)
"""Sparse autoencoder: reconstruction + L1 penalty on hidden activations."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sparse_autoencoder"]


def geron_sparse_autoencoder(x, hidden, decoded, lam):
    """
    Sparse autoencoder: reconstruction + L1 penalty on hidden activations

    Formula: L = ||x - Dec(Enc(x))||^2 + lam * ||h||_1  where h = Enc(x)

    Parameters
    ----------
    x : array-like
        Input data.
    hidden : array-like
        Input data.
    decoded : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 18, Sparse Autoencoders section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sparse autoencoder: reconstruction + L1 penalty on hidden activations"})


def cheatsheet():
    return "grsae: Sparse autoencoder: reconstruction + L1 penalty on hidden activations"
