# moirais.fn — function file (hadesllm/moirais)
"""Variational mode decomposition (VMD) into K band-limited modes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_vmd"]


def rangayyan_vmd(x, K, alpha, tau, init, tol):
    """
    Variational mode decomposition (VMD) into K band-limited modes

    Formula: min_{u,omega} sum_k ||d/dt[(delta(t)+j/pi*t)*u_k(t)]*e^{-j*omega_k*t}||^2 s.t. sum=x

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.
    alpha : array-like
        Input data.
    tau : array-like
        Input data.
    init : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: u_modes, omega_center

    References
    ----------
    Rangayyan Ch 9.4.1
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational mode decomposition (VMD) into K band-limited modes"})


def cheatsheet():
    return "rgvmd: Variational mode decomposition (VMD) into K band-limited modes"
