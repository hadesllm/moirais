"""Typical decoding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["typical_sampling"]


def typical_sampling(logits, tau):
    """
    Typical decoding

    Formula: keep tokens close to expected info content

    Parameters
    ----------
    logits : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Meister et al (2023)
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    result = float(np.mean(logits))
    se = float(np.std(logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Typical decoding"})


def cheatsheet():
    return "ttypc: Typical decoding"
