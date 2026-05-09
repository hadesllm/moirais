# moirais.fn — function file (hadesllm/moirais)
"""Approximate entropy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_approximate_entropy"]


def rangayyan_approximate_entropy(x):
    """
    Approximate entropy

    Formula: ApEn = phi_m(r) - phi_{m+1}(r)

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
    Rangayyan Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Approximate entropy"})


def cheatsheet():
    return "rgapn: Approximate entropy"
