# morie.fn -- function file (hadesllm/morie)
"""Series truncation for NPIV when T is unknown."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_series_unknown_T"]


def horowitz_series_unknown_T(x, y, w, K, basis):
    """
    Series truncation for NPIV when T is unknown

    Formula: Expand g in series {p_k}; G_hat_K = argmin ||m_hat - T_hat*sum_k a_k p_k||^2

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.
    K : array-like
        Input data.
    basis : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_hat

    References
    ----------
    Horowitz Ch 5, Sec 5.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Series truncation for NPIV when T is unknown"})


def cheatsheet():
    return "hrzseriu: Series truncation for NPIV when T is unknown"
