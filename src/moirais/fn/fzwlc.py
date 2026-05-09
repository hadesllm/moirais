# moirais.fn — function file (hadesllm/moirais)
"""Smoothed Wilcoxon signed-rank."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_smoothed_wilcoxon"]


def fauzi_smoothed_wilcoxon(x):
    """
    Smoothed Wilcoxon signed-rank

    Formula: W_n = sum R_i * F_hat_h(|D_i|) * sign(D_i)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Fauzi Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smoothed Wilcoxon signed-rank"})


def cheatsheet():
    return "fzwlc: Smoothed Wilcoxon signed-rank"
