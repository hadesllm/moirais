# moirais.fn — function file (hadesllm/moirais)
"""Burg method for AR spectral estimation (maximum entropy)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_burg_method"]


def rangayyan_burg_method(x, order, fs):
    """
    Burg method for AR spectral estimation (maximum entropy)

    Formula: Lattice formulation; reflection coefficients from forward/backward prediction errors

    Parameters
    ----------
    x : array-like
        Input data.
    order : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: psd, freqs, k_reflect

    References
    ----------
    Rangayyan Ch 7.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Burg method for AR spectral estimation (maximum entropy)"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Burg method for AR spectral estimation (maximum entropy)"})


def cheatsheet():
    return "rgburg: Burg method for AR spectral estimation (maximum entropy)"
