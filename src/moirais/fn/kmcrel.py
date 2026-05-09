# moirais.fn — function file (hadesllm/moirais)
"""RAGAS context relevance: fraction of retrieved context sentences that are useful."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_ragas_context_relevance"]


def kamath_ragas_context_relevance(context_sentences, relevance_labels):
    """
    RAGAS context relevance: fraction of retrieved context sentences that are useful

    Formula: ctxrel = |relevant_sentences| / |total_sentences_in_context|

    Parameters
    ----------
    context_sentences : array-like
        Input data.
    relevance_labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 7, Context Relevance section
    """
    context_sentences = np.atleast_1d(np.asarray(context_sentences, dtype=float))
    n = len(context_sentences)
    result = float(np.mean(context_sentences))
    se = float(np.std(context_sentences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RAGAS context relevance: fraction of retrieved context sentences that are useful"})


def cheatsheet():
    return "kmcrel: RAGAS context relevance: fraction of retrieved context sentences that are useful"
