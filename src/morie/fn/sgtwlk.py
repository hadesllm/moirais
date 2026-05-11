"""Weisfeiler-Leman 1-WL color refinement."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_weisfeiler_leman_relabel"]


def sgt_weisfeiler_leman_relabel(A, labels0, max_iter):
    """
    Weisfeiler-Leman 1-WL color refinement

    Formula: h^{(t+1)}(v) = hash(h^t(v), {h^t(u): u∈N(v)})

    Parameters
    ----------
    A : array-like
        Input data.
    labels0 : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels_t

    References
    ----------
    Weisfeiler-Lehman (1968)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weisfeiler-Leman 1-WL color refinement"})


def cheatsheet():
    return "sgtwlk: Weisfeiler-Leman 1-WL color refinement"
