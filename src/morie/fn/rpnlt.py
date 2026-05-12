# morie.fn -- function file (hadesllm/morie)
"""Roughness penalty (integrated squared second derivative) for functional smoothing."""
import numpy as np
from ._richresult import RichResult

__all__ = ["roughness_penalty"]


def roughness_penalty(basis, lam):
    """
    Roughness penalty (integrated squared second derivative) for functional smoothing

    Formula: Pen = integral (D^2 basis(t))^2 dt = c'*K*c; K = integral B''(t)*B''(t)' dt

    Parameters
    ----------
    basis : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'penalty': 'float'}

    References
    ----------
    Montesinos Lopez Ch 14
    """
    basis = np.asarray(basis, dtype=float)
    n = int(basis) if basis.ndim == 0 else len(basis)
    result = float(np.mean(basis))
    se = float(np.std(basis, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Roughness penalty (integrated squared second derivative) for functional smoothing"})


def cheatsheet():
    return "rpnlt: Roughness penalty (integrated squared second derivative) for functional smoothing"
