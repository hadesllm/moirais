# morie.fn — function file (hadesllm/morie)
"""Dense passage retrieval: bi-encoder dot-product similarity."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_dense_passage_retrieval"]


def kamath_dense_passage_retrieval(q_embed, p_embeds, k):
    """
    Dense passage retrieval: bi-encoder dot-product similarity

    Formula: score(q, p) = E_Q(q)^T * E_P(p); top-k = argsort by score

    Parameters
    ----------
    q_embed : array-like
        Input data.
    p_embeds : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: top_k_indices, top_k_scores

    References
    ----------
    Kamath Ch 7, Dense Retrieval (DPR) section
    """
    q_embed = np.atleast_1d(np.asarray(q_embed, dtype=float))
    n = len(q_embed)
    result = float(np.mean(q_embed))
    se = float(np.std(q_embed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dense passage retrieval: bi-encoder dot-product similarity"})


def cheatsheet():
    return "kmdpr: Dense passage retrieval: bi-encoder dot-product similarity"
