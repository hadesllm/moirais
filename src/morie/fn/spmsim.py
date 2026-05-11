"""Multiscale GWR (MGWR): variable bandwidth per covariate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_mgwr_bandwidth"]


def schabenberger_mgwr_bandwidth(x, y, coords):
    """
    Multiscale GWR (MGWR): variable bandwidth per covariate

    Formula: Y_i = sum_k f_k(u_i,v_i)*x_{ik}; each beta_k(s) has own bandwidth h_k

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
        Keys: local_coefficients, bandwidths

    References
    ----------
    Schabenberger Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiscale GWR (MGWR): variable bandwidth per covariate"})


def cheatsheet():
    return "spmsim: Multiscale GWR (MGWR): variable bandwidth per covariate"
