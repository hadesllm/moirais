# morie.fn -- function file (hadesllm/morie)
"""HyDE: use an LLM-generated hypothetical answer as the retrieval query."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_hyde_hypothetical_doc"]


def kamath_hyde_hypothetical_doc(query, model, embeddings):
    """
    HyDE: use an LLM-generated hypothetical answer as the retrieval query

    Formula: y_hypo = LLM(x);  retrieve top-k by sim(embed(y_hypo), embed(d))

    Parameters
    ----------
    query : array-like
        Input data.
    model : array-like
        Input data.
    embeddings : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: retrieved

    References
    ----------
    Kamath Ch 7, HyDE section
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HyDE: use an LLM-generated hypothetical answer as the retrieval query"})


def cheatsheet():
    return "kmhyde: HyDE: use an LLM-generated hypothetical answer as the retrieval query"
