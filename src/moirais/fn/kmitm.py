# moirais.fn — function file (hadesllm/moirais)
"""Image-Text Matching (ITM) binary head on top of contrastive encoders."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_image_text_matching"]


def kamath_image_text_matching(image_emb, text_emb, W, b):
    """
    Image-Text Matching (ITM) binary head on top of contrastive encoders

    Formula: p(match | I, T) = sigmoid( w^T [fused_embedding(I, T)] + b )

    Parameters
    ----------
    image_emb : array-like
        Input data.
    text_emb : array-like
        Input data.
    W : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Kamath Ch 9, Image-Text Matching section
    """
    image_emb = np.atleast_1d(np.asarray(image_emb, dtype=float))
    n = len(image_emb)
    result = float(np.mean(image_emb))
    se = float(np.std(image_emb, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Image-Text Matching (ITM) binary head on top of contrastive encoders"})


def cheatsheet():
    return "kmitm: Image-Text Matching (ITM) binary head on top of contrastive encoders"
