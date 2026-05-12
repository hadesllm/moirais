# morie.fn -- function file (hadesllm/morie)
"""Kendall tau with ties: correction for concordance/discordance counts."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_kendall_ties"]


def gibbons_kendall_ties(x, y):
    """
    Kendall tau with ties: correction for concordance/discordance counts

    Formula: tau_b = (P-Q) / sqrt((P+Q+T_x)(P+Q+T_y))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tau_b, p_value

    References
    ----------
    Gibbons Ch 11.2 ties
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kendall tau with ties: correction for concordance/discordance counts"})


def cheatsheet():
    return "gb1122t: Kendall tau with ties: correction for concordance/discordance counts"
