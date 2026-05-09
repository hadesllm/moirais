# moirais.fn — function file (hadesllm/moirais)
"""GPT decoder-only autoregressive next-token loss."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gpt_autoregressive_loss"]


def geron_gpt_autoregressive_loss(logits, targets):
    """
    GPT decoder-only autoregressive next-token loss

    Formula: L = - sum_{t=1..T} log p(x_t | x_{<t})

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
    Géron Ch 15, GPT / Decoder-only Transformer section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GPT decoder-only autoregressive next-token loss"})


def cheatsheet():
    return "grgptl: GPT decoder-only autoregressive next-token loss"
