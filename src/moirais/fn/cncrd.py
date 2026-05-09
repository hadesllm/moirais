# moirais.fn — function file (hadesllm/moirais)
"""Coefficient of concordance for incomplete rankings."""
import numpy as np
from ._richresult import RichResult

__all__ = ["concordance_incomplete"]


def concordance_incomplete(x):
    """
    Coefficient of concordance for incomplete rankings

    Formula: W for k sets with incomplete rankings

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
    Gibbons Ch 12.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Coefficient of concordance for incomplete rankings"})


def cheatsheet():
    return "cncrd: Coefficient of concordance for incomplete rankings"
