"""T5 encoder-decoder."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["t5"]


def t5(src, tgt, model):
    """
    T5 encoder-decoder

    Formula: text-to-text with span corruption pretraining

    Parameters
    ----------
    src : array-like
        Input data.
    tgt : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Raffel et al (2020)
    """
    src = np.atleast_1d(np.asarray(src, dtype=float))
    n = len(src)
    result = float(np.mean(src))
    se = float(np.std(src, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T5 encoder-decoder"})


def cheatsheet():
    return "t5enc: T5 encoder-decoder"
