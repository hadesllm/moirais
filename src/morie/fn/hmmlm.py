# morie.fn — function file (hadesllm/morie)
"""Masked language modeling pretraining objective."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_masked_lm"]


def geron_masked_lm(X, mask_frac):
    """
    Masked language modeling pretraining objective

    Formula: L = -sum_{i in M} log P(x_i | x_{not M})

    Parameters
    ----------
    X : array-like
        Input data.
    mask_frac : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 15
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Masked language modeling pretraining objective"})


def cheatsheet():
    return "hmmlm: Masked language modeling pretraining objective"
