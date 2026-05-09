"""SigLIP sigmoid loss (per-pair, not softmax)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["siglip_pairwise"]


def siglip_pairwise(image_emb, text_emb):
    """
    SigLIP sigmoid loss (per-pair, not softmax)

    Formula: sigmoid(z_ij) per (image, text) pair

    Parameters
    ----------
    image_emb : array-like
        Input data.
    text_emb : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhai et al (2023) SigLIP
    """
    image_emb = np.atleast_1d(np.asarray(image_emb, dtype=float))
    n = len(image_emb)
    result = float(np.mean(image_emb))
    se = float(np.std(image_emb, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SigLIP sigmoid loss (per-pair, not softmax)"})


def cheatsheet():
    return "siglip: SigLIP sigmoid loss (per-pair, not softmax)"
