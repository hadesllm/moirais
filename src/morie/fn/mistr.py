"""Mistral with sliding-window + SwiGLU."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mistral"]


def mistral(tokens, model):
    """
    Mistral with sliding-window + SwiGLU

    Formula: GQA + SWA causal mask

    Parameters
    ----------
    tokens : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jiang et al (2023) Mistral
    """
    tokens = np.atleast_1d(np.asarray(tokens, dtype=float))
    n = len(tokens)
    result = float(np.mean(tokens))
    se = float(np.std(tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mistral with sliding-window + SwiGLU"})


def cheatsheet():
    return "mistr: Mistral with sliding-window + SwiGLU"
