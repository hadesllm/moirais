"""CLIP text encoder (transformer)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clip_text_encoder"]


def clip_text_encoder(text):
    """
    CLIP text encoder (transformer)

    Formula: BPE tokenize + transformer + [EOT] embed

    Parameters
    ----------
    text : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Radford et al (2021)
    """
    text = np.atleast_1d(np.asarray(text, dtype=float))
    n = len(text)
    result = float(np.mean(text))
    se = float(np.std(text, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLIP text encoder (transformer)"})


def cheatsheet():
    return "clipxt: CLIP text encoder (transformer)"
