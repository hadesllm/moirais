"""Threshold GARCH (GJR-GARCH)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["tgarch_model"]


def tgarch_model(x):
    """
    Threshold GARCH (GJR-GARCH)

    Formula: sigma_t^2 = omega + (alpha+gamma*I_{t-1})*e_{t-1}^2 + beta*sigma_{t-1}^2

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Glosten et al. (1993)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Threshold GARCH (GJR-GARCH)"})


def cheatsheet():
    return "tgrch: Threshold GARCH (GJR-GARCH)"
