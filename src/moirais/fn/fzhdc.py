# moirais.fn — function file (hadesllm/moirais)
"""H-decomposition for asymptotic theory."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_h_decomposition"]


def fauzi_h_decomposition(x):
    """
    H-decomposition for asymptotic theory

    Formula: U-statistic decomposition into projections

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
    Fauzi Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "H-decomposition for asymptotic theory"})


def cheatsheet():
    return "fzhdc: H-decomposition for asymptotic theory"
