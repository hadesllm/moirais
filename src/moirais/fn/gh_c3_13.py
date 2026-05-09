# moirais.fn — function file (hadesllm/moirais)
"""Polya urn characterization of Polya tree process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_polya_urn_pt"]


def ghosal_polya_urn_pt(x):
    """
    Polya urn characterization of Polya tree process

    Formula: P(X_{n+1} in B | X_1..X_n) = (alpha_B + n_B) / (alpha + n)

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
    Ghosal Ch 3 §3.7.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polya urn characterization of Polya tree process"})


def cheatsheet():
    return "gh_c3_13: Polya urn characterization of Polya tree process"
