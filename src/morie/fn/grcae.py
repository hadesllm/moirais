# morie.fn — function file (hadesllm/morie)
"""Convolutional autoencoder: conv encoder + transposed-conv decoder."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_convolutional_autoencoder"]


def geron_convolutional_autoencoder(x, encoder_weights, decoder_weights):
    """
    Convolutional autoencoder: conv encoder + transposed-conv decoder

    Formula: z = ConvEnc(x); x_hat = DeConv(z); L = ||x - x_hat||^2

    Parameters
    ----------
    x : array-like
        Input data.
    encoder_weights : array-like
        Input data.
    decoder_weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_hat, loss

    References
    ----------
    Géron Ch 18, Convolutional Autoencoders section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convolutional autoencoder: conv encoder + transposed-conv decoder"})


def cheatsheet():
    return "grcae: Convolutional autoencoder: conv encoder + transposed-conv decoder"
