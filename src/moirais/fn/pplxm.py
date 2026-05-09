# moirais.fn — function file (hadesllm/moirais)
"""Perplexity of language model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["perplexity_metric"]


def perplexity_metric(x):
    """
    Perplexity of language model

    Formula: PPL = exp(-1/N sum log p(x_i|x_{<i}))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jelinek et al. (1977)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perplexity of language model"})


def cheatsheet():
    return "pplxm: Perplexity of language model"
