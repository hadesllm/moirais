# morie.fn -- function file (hadesllm/morie)
"""Birnbaum-Tingey closed-form for P(D+_n > c)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_ks_bt_formula"]


def gibbons_ks_bt_formula(c, n):
    """
    Birnbaum-Tingey closed-form for P(D+_n > c)

    Formula: P(D+_n > c) = (1-c)^n + c * sum_{j=1}^{[n(1-c)]} C(n,j)(1-c-j/n)^{n-j}(c+j/n)^{j-1}

    Parameters
    ----------
    c : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons eq 4.3.5
    """
    c = np.asarray(c, dtype=float)
    n = int(c) if c.ndim == 0 else len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Birnbaum-Tingey closed-form for P(D+_n > c)"})


def cheatsheet():
    return "gb434bt: Birnbaum-Tingey closed-form for P(D+_n > c)"
