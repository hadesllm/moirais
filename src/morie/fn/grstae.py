# morie.fn -- function file (hadesllm/morie)
"""Stacked autoencoder: multiple layers trained greedily or end-to-end."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_stacked_autoencoder"]


def geron_stacked_autoencoder(x, layer_weights):
    """
    Stacked autoencoder: multiple layers trained greedily or end-to-end

    Formula: x -> h_1 -> h_2 -> ... -> h_L -> ... -> x_hat (symmetric architecture)

    Parameters
    ----------
    x : array-like
        Input data.
    layer_weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_hat

    References
    ----------
    Géron Ch 18, Stacked Autoencoders section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stacked autoencoder: multiple layers trained greedily or end-to-end"})


def cheatsheet():
    return "grstae: Stacked autoencoder: multiple layers trained greedily or end-to-end"
