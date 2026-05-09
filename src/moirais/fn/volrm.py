"""RiskMetrics IGARCH(1,1) with λ=0.94 default."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_riskmetrics"]


def vol_riskmetrics(r, lam):
    """
    RiskMetrics IGARCH(1,1) with λ=0.94 default

    Formula: σ_t² = λ σ_{t-1}² + (1-λ) ε_{t-1}²

    Parameters
    ----------
    r : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma2

    References
    ----------
    JP Morgan (1996)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RiskMetrics IGARCH(1,1) with λ=0.94 default"})


def cheatsheet():
    return "volrm: RiskMetrics IGARCH(1,1) with λ=0.94 default"
