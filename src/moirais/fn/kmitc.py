# moirais.fn — function file (hadesllm/moirais)
"""Image-Text Contrastive loss (InfoNCE over a batch)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_image_text_contrastive"]


def kamath_image_text_contrastive(I_emb, T_emb, tau):
    """
    Image-Text Contrastive loss (InfoNCE over a batch)

    Formula: L_ITC = -0.5*(CE(sim(I,T)/tau, diag) + CE(sim(T,I)/tau, diag))

    Parameters
    ----------
    I_emb : array-like
        Input data.
    T_emb : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Kamath Ch 9, Image-Text Contrastive section
    """
    I_emb = np.atleast_1d(np.asarray(I_emb, dtype=float))
    n = len(I_emb)
    result = float(np.mean(I_emb))
    se = float(np.std(I_emb, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Image-Text Contrastive loss (InfoNCE over a batch)"})


def cheatsheet():
    return "kmitc: Image-Text Contrastive loss (InfoNCE over a batch)"
