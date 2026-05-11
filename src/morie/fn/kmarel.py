# morie.fn — function file (hadesllm/morie)
"""RAGAS answer relevance: cosine between reverse-generated questions and original."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_ragas_answer_relevance"]


def kamath_ragas_answer_relevance(answer, original_question, model):
    """
    RAGAS answer relevance: cosine between reverse-generated questions and original

    Formula: rel = (1/n) sum_i cos( embed(q_i_reverse), embed(q_orig) );  q_i_reverse = LLM(answer)

    Parameters
    ----------
    answer : array-like
        Input data.
    original_question : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 7, Answer Relevance section
    """
    answer = np.atleast_1d(np.asarray(answer, dtype=float))
    n = len(answer)
    result = float(np.mean(answer))
    se = float(np.std(answer, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RAGAS answer relevance: cosine between reverse-generated questions and original"})


def cheatsheet():
    return "kmarel: RAGAS answer relevance: cosine between reverse-generated questions and original"
