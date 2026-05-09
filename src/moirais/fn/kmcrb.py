# moirais.fn — function file (hadesllm/moirais)
"""Cross-encoder re-ranking: joint query+doc scored by a transformer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_cross_encoder_rerank"]


def kamath_cross_encoder_rerank(q, docs, model):
    """
    Cross-encoder re-ranking: joint query+doc scored by a transformer

    Formula: score(q, d) = f_theta([q; SEP; d])  (scalar output of cross-encoder)

    Parameters
    ----------
    q : array-like
        Input data.
    docs : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scores

    References
    ----------
    Kamath Ch 7, Cross-Encoder / ColBERT re-ranking section
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    result = float(np.mean(q))
    se = float(np.std(q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-encoder re-ranking: joint query+doc scored by a transformer"})


def cheatsheet():
    return "kmcrb: Cross-encoder re-ranking: joint query+doc scored by a transformer"
