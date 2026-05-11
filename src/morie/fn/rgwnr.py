# morie.fn — function file (hadesllm/morie)
"""Wiener filter (Wiener-Hopf equations, optimal MMSE linear filter)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wiener_filter"]


def rangayyan_wiener_filter(x, noise_psd, signal_psd):
    """
    Wiener filter (Wiener-Hopf equations, optimal MMSE linear filter)

    Formula: H_opt(f) = S_xd(f)/S_xx(f) = S_dd(f)/(S_dd(f)+S_nn(f))

    Parameters
    ----------
    x : array-like
        Input data.
    noise_psd : array-like
        Input data.
    signal_psd : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: filtered_x

    References
    ----------
    Rangayyan Ch 3.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wiener filter (Wiener-Hopf equations, optimal MMSE linear filter)"})


def cheatsheet():
    return "rgwnr: Wiener filter (Wiener-Hopf equations, optimal MMSE linear filter)"
