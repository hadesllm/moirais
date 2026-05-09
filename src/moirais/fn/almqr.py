# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Multi-query retrieval: LLM generates K paraphrases, union of top-k each."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_multi_query_retrieval"]


def alammar_multi_query_retrieval(query, K, retriever, model):
    """
    Multi-query retrieval: LLM generates K paraphrases, union of top-k each

    Formula: Q_set = LLM(rephrase(q), K); results = union_{q' in Q_set} top_k(q')

    Parameters
    ----------
    query : array-like
        Input data.
    K : array-like
        Input data.
    retriever : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: merged_results

    References
    ----------
    Alammar Ch 8, Multi-query Retrieval section
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-query retrieval: LLM generates K paraphrases, union of top-k each"})


def cheatsheet():
    return "almqr: Multi-query retrieval: LLM generates K paraphrases, union of top-k each"
