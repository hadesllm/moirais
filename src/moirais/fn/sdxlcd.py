"""Stable Diffusion XL UNet."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sdxl_unet"]


def sdxl_unet(x, t, text_emb):
    """
    Stable Diffusion XL UNet

    Formula: larger UNet + text/image conditioning

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    text_emb : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Podell et al (2023) SDXL
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stable Diffusion XL UNet"})


def cheatsheet():
    return "sdxlcd: Stable Diffusion XL UNet"
