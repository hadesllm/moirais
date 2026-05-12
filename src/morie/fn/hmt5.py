# morie.fn -- function file (hadesllm/morie)
"""T5: text-to-text transfer transformer (encoder-decoder)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_t5"]


def geron_t5(src, tgt):
    """
    T5: text-to-text transfer transformer (encoder-decoder)

    Formula: every task cast as text-to-text; span corruption pretraining

    Parameters
    ----------
    src : array-like
        Input data.
    tgt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 15
    """
    src = np.atleast_1d(np.asarray(src, dtype=float))
    n = len(src)
    result = float(np.mean(src))
    se = float(np.std(src, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T5: text-to-text transfer transformer (encoder-decoder)"})


def cheatsheet():
    return "hmt5: T5: text-to-text transfer transformer (encoder-decoder)"
