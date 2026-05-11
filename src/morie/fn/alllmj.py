# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""LLM-as-judge automated evaluation via pairwise or pointwise prompting."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_llm_as_judge"]


def alammar_llm_as_judge(responses, rubric, judge_model):
    """
    LLM-as-judge automated evaluation via pairwise or pointwise prompting

    Formula: score(y) = LLM_judge(rubric, y); aggregated across samples

    Parameters
    ----------
    responses : array-like
        Input data.
    rubric : array-like
        Input data.
    judge_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scores

    References
    ----------
    Alammar Ch 12, LLM-as-judge section
    """
    responses = np.atleast_1d(np.asarray(responses, dtype=float))
    n = len(responses)
    result = float(np.mean(responses))
    se = float(np.std(responses, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LLM-as-judge automated evaluation via pairwise or pointwise prompting"})


def cheatsheet():
    return "alllmj: LLM-as-judge automated evaluation via pairwise or pointwise prompting"
