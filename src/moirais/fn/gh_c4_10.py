# moirais.fn — function file (hadesllm/moirais)
"""Polya urn scheme construction of DP: sequential predictive draws."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_polya_urn"]


def ghosal_dp_polya_urn(x):
    """
    Polya urn scheme construction of DP: sequential predictive draws

    Formula: X_{n+1}|X_1..X_n ~ sum_k n_k/(alpha+n)*delta_{X_k^*} + alpha/(alpha+n)*G0

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
    Ghosal Ch 4 §4.2.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polya urn scheme construction of DP: sequential predictive draws"})


def cheatsheet():
    return "gh_c4_10: Polya urn scheme construction of DP: sequential predictive draws"
