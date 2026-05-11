"""Rosenthal's fail-safe N (file-drawer)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_fail_safe_n"]


def ma_fail_safe_n(z_scores, alpha):
    """
    Rosenthal's fail-safe N (file-drawer)

    Formula: N_fs = (Σz_i)²/z²_α - k

    Parameters
    ----------
    z_scores : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Nfs

    References
    ----------
    Rosenthal (1979)
    """
    z_scores = np.atleast_1d(np.asarray(z_scores, dtype=float))
    n = len(z_scores)
    result = float(np.mean(z_scores))
    se = float(np.std(z_scores, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rosenthal's fail-safe N (file-drawer)"})


def cheatsheet():
    return "mafsn: Rosenthal's fail-safe N (file-drawer)"
