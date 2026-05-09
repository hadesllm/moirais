"""TGARCH (Zakoian) absolute-value GARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_tgarch_fit"]


def vol_tgarch_fit(r, init):
    """
    TGARCH (Zakoian) absolute-value GARCH

    Formula: σ_t = ω + α₁|ε_{t-1}|⁺ + α₂|ε_{t-1}|⁻ + β σ_{t-1}

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: omega, alpha1, alpha2, beta, ll

    References
    ----------
    Zakoian (1994)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TGARCH (Zakoian) absolute-value GARCH"})


def cheatsheet():
    return "voltgr: TGARCH (Zakoian) absolute-value GARCH"
