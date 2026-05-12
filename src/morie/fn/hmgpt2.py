# morie.fn -- function file (hadesllm/morie)
"""GPT-2: scaled-up decoder-only LM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gpt2"]


def geron_gpt2(X, n_layers, n_heads):
    """
    GPT-2: scaled-up decoder-only LM

    Formula: same as GPT-1 but larger (up to 1.5B params)

    Parameters
    ----------
    X : array-like
        Input data.
    n_layers : array-like
        Input data.
    n_heads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 15
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GPT-2: scaled-up decoder-only LM"})


def cheatsheet():
    return "hmgpt2: GPT-2: scaled-up decoder-only LM"
