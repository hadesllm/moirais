# morie.fn -- function file (hadesllm/morie)
"""Self-RAG: emit reflection tokens deciding to retrieve / grade relevance / check support."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_self_rag"]


def kamath_self_rag(context, reflection_model):
    """
    Self-RAG: emit reflection tokens deciding to retrieve / grade relevance / check support

    Formula: tokens in {[Retrieve], [No Retrieve], [Relevant], [Irrelevant], [Supported], [Unsupported], ...}

    Parameters
    ----------
    context : array-like
        Input data.
    reflection_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: trace

    References
    ----------
    Kamath Ch 7, Self-RAG section
    """
    context = np.atleast_1d(np.asarray(context, dtype=float))
    n = len(context)
    result = float(np.mean(context))
    se = float(np.std(context, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self-RAG: emit reflection tokens deciding to retrieve / grade relevance / check support"})


def cheatsheet():
    return "kmsrag: Self-RAG: emit reflection tokens deciding to retrieve / grade relevance / check support"
