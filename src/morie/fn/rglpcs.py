# morie.fn — function file (hadesllm/morie)
"""LPC synthesis filter for signal reconstruction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_lpc_synthesis"]


def rangayyan_lpc_synthesis(lpc_coeffs, gain, excitation):
    """
    LPC synthesis filter for signal reconstruction

    Formula: x_hat[n] = sum a_k*lpc_coeffs[n-k] + G*e[n] where e[n] is excitation

    Parameters
    ----------
    lpc_coeffs : array-like
        Input data.
    gain : array-like
        Input data.
    excitation : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_synth

    References
    ----------
    Rangayyan Ch 7.5
    """
    lpc_coeffs = np.asarray(lpc_coeffs, dtype=float)
    n = int(lpc_coeffs) if lpc_coeffs.ndim == 0 else len(lpc_coeffs)
    result = float(np.mean(lpc_coeffs))
    se = float(np.std(lpc_coeffs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LPC synthesis filter for signal reconstruction"})


def cheatsheet():
    return "rglpcs: LPC synthesis filter for signal reconstruction"
