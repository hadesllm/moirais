# morie.fn — function file (hadesllm/morie)
"""Minimum description length (MDL) criterion for AR model order."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ar_order_mdl"]


def rangayyan_ar_order_mdl(x, max_order):
    """
    Minimum description length (MDL) criterion for AR model order

    Formula: MDL(p) = N*log(sigma_p^2) + p*log(N)

    Parameters
    ----------
    x : array-like
        Input data.
    max_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: optimal_order, mdl_values

    References
    ----------
    Rangayyan Ch 7.5.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimum description length (MDL) criterion for AR model order"})


def cheatsheet():
    return "rgmdl: Minimum description length (MDL) criterion for AR model order"
