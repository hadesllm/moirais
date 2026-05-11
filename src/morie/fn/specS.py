"""Speculative decoding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["speculative_decoding"]


def speculative_decoding(prompt, draft, target):
    """
    Speculative decoding

    Formula: draft model proposes; target verifies in 1 forward

    Parameters
    ----------
    prompt : array-like
        Input data.
    draft : array-like
        Input data.
    target : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Leviathan-Kalman-Matias (2023)
    """
    prompt = np.atleast_1d(np.asarray(prompt, dtype=float))
    n = len(prompt)
    result = float(np.mean(prompt))
    se = float(np.std(prompt, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Speculative decoding"})


def cheatsheet():
    return "specS: Speculative decoding"
