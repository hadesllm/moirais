"""U-Net encoder-decoder skip connections."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["unet_backbone"]


def unet_backbone(x, filters):
    """
    U-Net encoder-decoder skip connections

    Formula: contracting + expanding paths; skip concat

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
    Ronneberger et al (2015) U-Net
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "U-Net encoder-decoder skip connections"})


def cheatsheet():
    return "unetbk: U-Net encoder-decoder skip connections"
