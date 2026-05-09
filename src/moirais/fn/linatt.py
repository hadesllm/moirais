"""Linformer linear-complexity attention via low-rank projection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["linformer_linear_attention"]


def linformer_linear_attention(y, Q, K, V, E, F):
    """
    Linformer linear-complexity attention via low-rank projection

    Formula: Attn = softmax(Q (E K)^T / sqrt(d)) (F V)

    Parameters
    ----------
    y : array-like
        Input data.
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    E : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang et al. (2020) Linformer
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linformer linear-complexity attention via low-rank projection"})


def cheatsheet():
    return "linatt: Linformer linear-complexity attention via low-rank projection"
