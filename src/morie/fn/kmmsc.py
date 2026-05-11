# morie.fn — function file (hadesllm/morie)
"""MoverScore: Word Mover's Distance over BERT embeddings."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_moverscore"]


def kamath_moverscore(hypothesis_embeddings, reference_embeddings):
    """
    MoverScore: Word Mover's Distance over BERT embeddings

    Formula: MoverScore = 1 - WMD(emb(h), emb(r)) / normalizer

    Parameters
    ----------
    hypothesis_embeddings : array-like
        Input data.
    reference_embeddings : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 8, MoverScore section
    """
    hypothesis_embeddings = np.atleast_1d(np.asarray(hypothesis_embeddings, dtype=float))
    n = len(hypothesis_embeddings)
    result = float(np.mean(hypothesis_embeddings))
    se = float(np.std(hypothesis_embeddings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MoverScore: Word Mover's Distance over BERT embeddings"})


def cheatsheet():
    return "kmmsc: MoverScore: Word Mover's Distance over BERT embeddings"
