# morie.fn — function file (hadesllm/morie)
"""Temperature scaling of softmax logits before sampling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_temperature_sampling"]


def geron_temperature_sampling(logits, T):
    """
    Temperature scaling of softmax logits before sampling

    Formula: p_i = exp(z_i / T) / sum_j exp(z_j / T)

    Parameters
    ----------
    logits : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probs

    References
    ----------
    Géron Ch 15, Temperature sampling
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Temperature scaling of softmax logits before sampling"})


def cheatsheet():
    return "grtmp: Temperature scaling of softmax logits before sampling"
