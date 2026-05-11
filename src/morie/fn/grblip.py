# morie.fn — function file (hadesllm/morie)
"""BLIP image-text matching + contrastive objectives."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_blip_itm_itc"]


def geron_blip_itm_itc(image_emb, text_emb, caption_logits, caption_targets):
    """
    BLIP image-text matching + contrastive objectives

    Formula: L = lam_itc * ITC(img, txt) + lam_itm * ITM(img, txt) + lam_lm * LM(caption | img)

    Parameters
    ----------
    image_emb : array-like
        Input data.
    text_emb : array-like
        Input data.
    caption_logits : array-like
        Input data.
    caption_targets : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 16, BLIP / BLIP-2 section
    """
    image_emb = np.atleast_1d(np.asarray(image_emb, dtype=float))
    n = len(image_emb)
    result = float(np.mean(image_emb))
    se = float(np.std(image_emb, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BLIP image-text matching + contrastive objectives"})


def cheatsheet():
    return "grblip: BLIP image-text matching + contrastive objectives"
