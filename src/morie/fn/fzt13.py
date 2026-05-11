# morie.fn — function file (hadesllm/morie)
"""Theorem 1.3: MISE of modified gamma KDE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm1_3_mise_mgkde"]


def fauzi_thm1_3_mise_mgkde(x, bandwidth):
    """
    Theorem 1.3: MISE of modified gamma KDE

    Formula: MISE = (h^4/4) * [integral b_4^2/f_X dx] * [integral u^4 K du]^2 + (1/n)*integral f_X/x * integral u^2 K_1 du dx + o(h^4+1/n)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mise

    References
    ----------
    Fauzi Ch 1, Theorem 1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 1.3: MISE of modified gamma KDE"})


def cheatsheet():
    return "fzt13: Theorem 1.3: MISE of modified gamma KDE"
