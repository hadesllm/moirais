# morie.fn — function file (hadesllm/morie)
"""CLIP contrastive image-text loss (symmetric InfoNCE over a batch)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_clip_contrastive_loss"]


def geron_clip_contrastive_loss(image_embeddings, text_embeddings, tau):
    """
    CLIP contrastive image-text loss (symmetric InfoNCE over a batch)

    Formula: L = 0.5*(CE(sim(I,T)/tau, labels) + CE(sim(T,I)/tau, labels)); labels=diag

    Parameters
    ----------
    image_embeddings : array-like
        Input data.
    text_embeddings : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 16, CLIP section
    """
    image_embeddings = np.atleast_1d(np.asarray(image_embeddings, dtype=float))
    n = len(image_embeddings)
    result = float(np.mean(image_embeddings))
    se = float(np.std(image_embeddings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLIP contrastive image-text loss (symmetric InfoNCE over a batch)"})


def cheatsheet():
    return "grclp: CLIP contrastive image-text loss (symmetric InfoNCE over a batch)"
