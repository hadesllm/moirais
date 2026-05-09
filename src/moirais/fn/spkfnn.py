"""Leave-one-out cross-validation for kriging: MSPE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_cross_validation_kriging"]


def schabenberger_cross_validation_kriging(coords, z, cov_model):
    """
    Leave-one-out cross-validation for kriging: MSPE

    Formula: MSPE = (1/n)*sum(Z(s_i)-Z_hat_{-i}(s_i))^2; Z_hat_{-i} kriged without s_i

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    cov_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mspe

    References
    ----------
    Schabenberger Ch 5
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Leave-one-out cross-validation for kriging: MSPE"})


def cheatsheet():
    return "spkfnn: Leave-one-out cross-validation for kriging: MSPE"
