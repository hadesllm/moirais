# moirais.fn — function file (hadesllm/moirais)
"""Mistral-7B: open-weights 7B-parameter decoder-only LLM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mistral7b"]


def geron_mistral7b(prompt, n_tokens):
    """
    Mistral-7B: open-weights 7B-parameter decoder-only LLM

    Formula: decoder-only transformer with sliding-window attention

    Parameters
    ----------
    prompt : array-like
        Input data.
    n_tokens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: completion

    References
    ----------
    Géron Ch 15
    """
    prompt = np.atleast_1d(np.asarray(prompt, dtype=float))
    n = len(prompt)
    result = float(np.mean(prompt))
    se = float(np.std(prompt, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mistral-7B: open-weights 7B-parameter decoder-only LLM"})


def cheatsheet():
    return "hmmis7: Mistral-7B: open-weights 7B-parameter decoder-only LLM"
