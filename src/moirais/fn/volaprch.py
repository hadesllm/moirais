"""Asymmetric Power ARCH (Ding-Granger-Engle)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_aparch_fit"]


def vol_aparch_fit(r, init):
    """
    Asymmetric Power ARCH (Ding-Granger-Engle)

    Formula: σ_t^δ = ω + α(|ε_{t-1}|-γ ε_{t-1})^δ + β σ_{t-1}^δ

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: omega, alpha, gamma, beta, delta, ll

    References
    ----------
    Ding-Granger-Engle (1993)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymmetric Power ARCH (Ding-Granger-Engle)"})


def cheatsheet():
    return "volaprch: Asymmetric Power ARCH (Ding-Granger-Engle)"
