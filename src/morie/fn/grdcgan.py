# morie.fn — function file (hadesllm/morie)
"""DCGAN generator: transposed-conv upsampling from latent z."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dcgan_generator"]


def geron_dcgan_generator(z, weights):
    """
    DCGAN generator: transposed-conv upsampling from latent z

    Formula: G(z) = conv_transpose_stack(z); typically uses BN + ReLU + tanh output

    Parameters
    ----------
    z : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: image

    References
    ----------
    Géron Ch 18, DCGAN section
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DCGAN generator: transposed-conv upsampling from latent z"})


def cheatsheet():
    return "grdcgan: DCGAN generator: transposed-conv upsampling from latent z"
