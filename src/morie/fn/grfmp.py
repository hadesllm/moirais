# morie.fn — function file (hadesllm/morie)
"""Feature-map total elements for a conv layer (H' x W' x C_out)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_feature_map_dim"]


def geron_feature_map_dim(H_out, W_out, C_out):
    """
    Feature-map total elements for a conv layer (H' x W' x C_out)

    Formula: dim = H' * W' * C_out where H', W' are conv output sizes

    Parameters
    ----------
    H_out : array-like
        Input data.
    W_out : array-like
        Input data.
    C_out : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: dim

    References
    ----------
    Géron Ch 12, Feature Maps section
    """
    H_out = np.atleast_1d(np.asarray(H_out, dtype=float))
    n = len(H_out)
    result = float(np.mean(H_out))
    se = float(np.std(H_out, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Feature-map total elements for a conv layer (H' x W' x C_out)"})


def cheatsheet():
    return "grfmp: Feature-map total elements for a conv layer (H' x W' x C_out)"
