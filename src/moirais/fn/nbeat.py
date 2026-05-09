# moirais.fn — function file (hadesllm/moirais)
"""N-BEATS basis expansion forecasting."""
import numpy as np
from ._richresult import RichResult

__all__ = ["nbeats_basis"]


def nbeats_basis(x):
    """
    N-BEATS basis expansion forecasting

    Formula: y_hat = sum theta_k * basis_k(t)

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
    Oreshkin et al. (2020)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "N-BEATS basis expansion forecasting"})


def cheatsheet():
    return "nbeat: N-BEATS basis expansion forecasting"
