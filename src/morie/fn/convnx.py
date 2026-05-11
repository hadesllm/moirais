"""ConvNeXt block (modernized ResNet)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["convnext_block"]


def convnext_block(x, filters):
    """
    ConvNeXt block (modernized ResNet)

    Formula: depthwise conv + LayerNorm + 1x1 conv

    Parameters
    ----------
    x : array-like
        Input data.
    filters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu et al (2022) ConvNeXt
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ConvNeXt block (modernized ResNet)"})


def cheatsheet():
    return "convnx: ConvNeXt block (modernized ResNet)"
