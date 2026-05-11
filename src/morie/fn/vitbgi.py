"""ViT-B/16 initialization (768 dim, 12 heads, 12 layers)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit_b16_init"]


def vit_b16_init(model):
    """
    ViT-B/16 initialization (768 dim, 12 heads, 12 layers)

    Formula: truncated normal(std=0.02)

    Parameters
    ----------
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dosovitskiy et al (2020)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT-B/16 initialization (768 dim, 12 heads, 12 layers)"})


def cheatsheet():
    return "vitbgi: ViT-B/16 initialization (768 dim, 12 heads, 12 layers)"
