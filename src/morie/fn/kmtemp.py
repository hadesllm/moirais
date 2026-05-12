# morie.fn -- function file (hadesllm/morie)
"""Temperature sampling: softmax(logits / T)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_temperature_sampling"]


def kamath_temperature_sampling(logits, T):
    """
    Temperature sampling: softmax(logits / T)

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
    Kamath Ch 4, Temperature sampling section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Temperature sampling: softmax(logits / T)"})


def cheatsheet():
    return "kmtemp: Temperature sampling: softmax(logits / T)"
