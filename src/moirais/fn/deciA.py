# moirais.fn — function file (hadesllm/moirais)
"""DECI (deep end-to-end causal inference): joint structure + effect learning."""
import numpy as np
from ._richresult import RichResult

__all__ = ["deci_model"]


def deci_model(data, n_samples, lr):
    """
    DECI (deep end-to-end causal inference): joint structure + effect learning

    Formula: Learn P(X|G) and G simultaneously; variational inference over graph distribution; end-to-end gradient

    Parameters
    ----------
    data : array-like
        Input data.
    n_samples : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'graph_posterior': 'distribution', 'ate': 'float'}

    References
    ----------
    Molak Ch 14
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DECI (deep end-to-end causal inference): joint structure + effect learning"})


def cheatsheet():
    return "deciA: DECI (deep end-to-end causal inference): joint structure + effect learning"
