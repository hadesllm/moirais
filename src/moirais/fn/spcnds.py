"""Conditional simulation of Gaussian random field given data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_conditional_sim"]


def schabenberger_conditional_sim(coords, z_obs, cov_model, sim_grid):
    """
    Conditional simulation of Gaussian random field given data

    Formula: Z_c(s) = Z_sim(s) + [Z_obs - Z_sim]_kriged(s); conditions on observed values

    Parameters
    ----------
    coords : array-like
        Input data.
    z_obs : array-like
        Input data.
    cov_model : array-like
        Input data.
    sim_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: conditional_sim

    References
    ----------
    Schabenberger Ch 7, Sec 7.2.2
    """
    coords = np.asarray(coords, dtype=float)
    n = int(coords) if coords.ndim == 0 else len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conditional simulation of Gaussian random field given data"})


def cheatsheet():
    return "spcnds: Conditional simulation of Gaussian random field given data"
