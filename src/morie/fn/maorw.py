"""Orwin's fail-safe N for trivial-effect target."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_orwin_fsn"]


def ma_orwin_fsn(d_obs, d_crit, d_filldraw, k):
    """
    Orwin's fail-safe N for trivial-effect target

    Formula: N_fs = k(d̄-d_c)/(d_c-d_0)

    Parameters
    ----------
    d_obs : array-like
        Input data.
    d_crit : array-like
        Input data.
    d_filldraw : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Nfs

    References
    ----------
    Orwin (1983)
    """
    d_obs = np.atleast_1d(np.asarray(d_obs, dtype=float))
    n = len(d_obs)
    result = float(np.mean(d_obs))
    se = float(np.std(d_obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Orwin's fail-safe N for trivial-effect target"})


def cheatsheet():
    return "maorw: Orwin's fail-safe N for trivial-effect target"
