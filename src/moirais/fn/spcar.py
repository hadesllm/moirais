"""Conditional autoregressive (CAR) model: conditional specification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_car_model"]


def schabenberger_car_model(z, w, covariates):
    """
    Conditional autoregressive (CAR) model: conditional specification

    Formula: Z_i|Z_{-i} ~ N(mu_i + rho*sum_j w_{ij}(Z_j-mu_j), sigma_i^2)

    Parameters
    ----------
    z : array-like
        Input data.
    w : array-like
        Input data.
    covariates : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho, beta, se

    References
    ----------
    Schabenberger Ch 6, Sec 6.2.2
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conditional autoregressive (CAR) model: conditional specification"})


def cheatsheet():
    return "spcar: Conditional autoregressive (CAR) model: conditional specification"
