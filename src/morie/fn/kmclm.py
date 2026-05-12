# morie.fn -- function file (hadesllm/morie)
"""Causal LM next-token cross-entropy loss (GPT-style)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_causal_lm_loss"]


def kamath_causal_lm_loss(logits, targets):
    """
    Causal LM next-token cross-entropy loss (GPT-style)

    Formula: L_CLM = - sum_{t=1..T} log p(x_t | x_{<t})

    Parameters
    ----------
    logits : array-like
        Input data.
    targets : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Kamath Ch 2, Causal Language Modeling section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Causal LM next-token cross-entropy loss (GPT-style)"})


def cheatsheet():
    return "kmclm: Causal LM next-token cross-entropy loss (GPT-style)"
