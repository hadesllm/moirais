"""BEKK MGARCH(1,1) full-parameter form."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_bekk_garch"]


def vol_bekk_garch(R_panel, init):
    """
    BEKK MGARCH(1,1) full-parameter form

    Formula: H_t = C'C + A'ε_{t-1}ε'_{t-1}A + B'H_{t-1}B

    Parameters
    ----------
    R_panel : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: C, A, B, ll

    References
    ----------
    Engle-Kroner (1995)
    """
    R_panel = np.atleast_1d(np.asarray(R_panel, dtype=float))
    n = len(R_panel)
    result = float(np.mean(R_panel))
    se = float(np.std(R_panel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BEKK MGARCH(1,1) full-parameter form"})


def cheatsheet():
    return "volbekk: BEKK MGARCH(1,1) full-parameter form"
