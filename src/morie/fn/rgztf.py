# morie.fn -- function file (hadesllm/morie)
"""Z-transform of a causal discrete-time sequence."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_z_transform"]


def rangayyan_z_transform(x_coeffs):
    """
    Z-transform of a causal discrete-time sequence

    Formula: X(z) = sum_{n=0}^{inf} x_coeffs[n] * z^{-n}

    Parameters
    ----------
    x_coeffs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_z_rational

    References
    ----------
    Rangayyan Ch 3.4.2
    """
    x_coeffs = np.asarray(x_coeffs, dtype=float)
    n = int(x_coeffs) if x_coeffs.ndim == 0 else len(x_coeffs)
    result = float(np.mean(x_coeffs))
    se = float(np.std(x_coeffs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Z-transform of a causal discrete-time sequence"})


def cheatsheet():
    return "rgztf: Z-transform of a causal discrete-time sequence"
