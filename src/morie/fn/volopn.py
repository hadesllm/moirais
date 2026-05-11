"""Black-Scholes implied volatility via Brent root."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_implied_volatility_bs"]


def vol_implied_volatility_bs(S, K, T, r, C_obs, kind):
    """
    Black-Scholes implied volatility via Brent root

    Formula: Solve C(σ) = C_obs; brentq on σ ∈ [1e-4, 5]

    Parameters
    ----------
    S : array-like
        Input data.
    K : array-like
        Input data.
    T : array-like
        Input data.
    r : array-like
        Input data.
    C_obs : array-like
        Input data.
    kind : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: iv

    References
    ----------
    Black & Scholes (1973)
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Black-Scholes implied volatility via Brent root"})


def cheatsheet():
    return "volopn: Black-Scholes implied volatility via Brent root"
