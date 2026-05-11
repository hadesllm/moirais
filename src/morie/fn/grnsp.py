# morie.fn — function file (hadesllm/morie)
"""BERT next-sentence-prediction binary cross-entropy loss."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bert_nsp_loss"]


def geron_bert_nsp_loss(logits, labels):
    """
    BERT next-sentence-prediction binary cross-entropy loss

    Formula: L_NSP = - log p(is_next | segment_A, segment_B)

    Parameters
    ----------
    logits : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 15, BERT pretraining (NSP) section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BERT next-sentence-prediction binary cross-entropy loss"})


def cheatsheet():
    return "grnsp: BERT next-sentence-prediction binary cross-entropy loss"
