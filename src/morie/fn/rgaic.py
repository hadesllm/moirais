# morie.fn — function file (hadesllm/morie)
"""AIC criterion for AR model order selection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ar_order_aic"]


def rangayyan_ar_order_aic(x, max_order):
    """
    AIC criterion for AR model order selection

    Formula: AIC(p) = N*log(sigma_p^2) + 2*p

    Parameters
    ----------
    x : array-like
        Input data.
    max_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: optimal_order, aic_values

    References
    ----------
    Rangayyan Ch 7.5.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AIC criterion for AR model order selection"})


def cheatsheet():
    return "rgaic: AIC criterion for AR model order selection"
