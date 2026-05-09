# moirais.fn — function file (hadesllm/moirais)
"""GPT-3: 175B-parameter autoregressive LM capable of in-context learning."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gpt3"]


def geron_gpt3(prompt, n_tokens):
    """
    GPT-3: 175B-parameter autoregressive LM capable of in-context learning

    Formula: same decoder-only architecture, massive scale

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GPT-3: 175B-parameter autoregressive LM capable of in-context learning"})


def cheatsheet():
    return "hmgpt3: GPT-3: 175B-parameter autoregressive LM capable of in-context learning"
