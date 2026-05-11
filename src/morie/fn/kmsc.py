# morie.fn — function file (hadesllm/morie)
"""Self-consistency decoding: sample N CoT traces and majority-vote the answer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_self_consistency"]


def kamath_self_consistency(samples):
    """
    Self-consistency decoding: sample N CoT traces and majority-vote the answer

    Formula: y_hat = mode_{i=1..N} parse(sample_i);  sample_i ~ LLM(CoT prompt)

    Parameters
    ----------
    samples : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat, confidence

    References
    ----------
    Kamath Ch 4, Self-Consistency section
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = len(samples)
    result = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self-consistency decoding: sample N CoT traces and majority-vote the answer"})


def cheatsheet():
    return "kmsc: Self-consistency decoding: sample N CoT traces and majority-vote the answer"
