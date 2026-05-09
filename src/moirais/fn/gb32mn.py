# moirais.fn — function file (hadesllm/moirais)
"""Mean of total number of runs under null hypothesis of randomness."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_runs_mean"]


def gibbons_runs_mean(n1, n2):
    """
    Mean of total number of runs under null hypothesis of randomness

    Formula: E(R) = 1 + 2*n1*n2 / (n1+n2)

    Parameters
    ----------
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean

    References
    ----------
    Gibbons eq 3.2.6
    """
    n1 = np.asarray(n1, dtype=float)
    n = int(n1) if n1.ndim == 0 else len(n1)
    result = float(np.mean(n1))
    se = float(np.std(n1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean of total number of runs under null hypothesis of randomness"})


def cheatsheet():
    return "gb32mn: Mean of total number of runs under null hypothesis of randomness"
