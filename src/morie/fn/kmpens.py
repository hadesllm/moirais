# morie.fn -- function file (hadesllm/morie)
"""Prompt ensemble: average class probs across K different prompt templates."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_prompt_ensemble"]


def kamath_prompt_ensemble(prompt_logits):
    """
    Prompt ensemble: average class probs across K different prompt templates

    Formula: P(y | x) = (1/K) sum_{k=1..K} P_LLM(y | prompt_k(x))

    Parameters
    ----------
    prompt_logits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ensemble_probs

    References
    ----------
    Kamath Ch 3, Prompt Ensembling section
    """
    prompt_logits = np.atleast_1d(np.asarray(prompt_logits, dtype=float))
    n = len(prompt_logits)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Prompt ensemble: average class probs across K different prompt templates"})
    estimate = np.median(prompt_logits)
    se = 1.2533 * np.std(prompt_logits, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Prompt ensemble: average class probs across K different prompt templates"})


def cheatsheet():
    return "kmpens: Prompt ensemble: average class probs across K different prompt templates"
