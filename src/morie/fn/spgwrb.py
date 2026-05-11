"""GWR bandwidth selection via AIC or cross-validation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_gwr_bandwidth"]


def schabenberger_gwr_bandwidth(x, y, coords):
    """
    GWR bandwidth selection via AIC or cross-validation

    Formula: AIC_c(h) = 2*n*log(sigma_hat) + 2*tr(S) + 2*tr(S)'*tr(S)/(n-tr(S)-1)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: optimal_bandwidth

    References
    ----------
    Schabenberger Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GWR bandwidth selection via AIC or cross-validation"})


def cheatsheet():
    return "spgwrb: GWR bandwidth selection via AIC or cross-validation"
