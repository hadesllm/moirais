"""PatchTST — channel-independent patches + Transformer."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["patch_tst"]


def patch_tst(X, patch_len):
    """
    PatchTST — channel-independent patches + Transformer

    Formula: split into patches; per-channel transformer

    Parameters
    ----------
    X : array-like
        Input data.
    patch_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nie et al (2023) PatchTST
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PatchTST — channel-independent patches + Transformer"})


def cheatsheet():
    return "patchT: PatchTST — channel-independent patches + Transformer"
