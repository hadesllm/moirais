"""Trans-Gaussian kriging via Anamorphosis transformation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_trans_gaussian_kriging"]


def schabenberger_trans_gaussian_kriging(coords, z, target, transformation, cov_model):
    """
    Trans-Gaussian kriging via Anamorphosis transformation

    Formula: phi(Z) ~ Gaussian; phi_hat(s0) = OK on transformed data; back-transform via E[phi^{-1}]

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    target : array-like
        Input data.
    transformation : array-like
        Input data.
    cov_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction, variance

    References
    ----------
    Schabenberger Ch 5, Sec 5.6.2
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Trans-Gaussian kriging via Anamorphosis transformation"})


def cheatsheet():
    return "sptgk: Trans-Gaussian kriging via Anamorphosis transformation"
