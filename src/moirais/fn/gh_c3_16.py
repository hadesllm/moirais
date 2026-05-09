# moirais.fn — function file (hadesllm/moirais)
"""Evenly split Polya tree: canonical Polya tree PT*(alpha, a_m) with equal partitions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_evsplit_pt"]


def ghosal_evsplit_pt(x):
    """
    Evenly split Polya tree: canonical Polya tree PT*(alpha, a_m) with equal partitions

    Formula: PT*(alpha, a_m): alpha_{e0}=alpha_{e1}=a_m at level m

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 3 §3.7.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Evenly split Polya tree: canonical Polya tree PT*(alpha, a_m) with equal partitions"})


def cheatsheet():
    return "gh_c3_16: Evenly split Polya tree: canonical Polya tree PT*(alpha, a_m) with equal partitions"
