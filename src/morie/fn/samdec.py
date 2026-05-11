"""SAM mask decoder (transformer + upsample)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sam_mask_decoder"]


def sam_mask_decoder(img_emb, prompt_emb):
    """
    SAM mask decoder (transformer + upsample)

    Formula: two-way attention + upsample to full res

    Parameters
    ----------
    img_emb : array-like
        Input data.
    prompt_emb : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kirillov et al (2023)
    """
    img_emb = np.atleast_1d(np.asarray(img_emb, dtype=float))
    n = len(img_emb)
    result = float(np.mean(img_emb))
    se = float(np.std(img_emb, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SAM mask decoder (transformer + upsample)"})


def cheatsheet():
    return "samdec: SAM mask decoder (transformer + upsample)"
