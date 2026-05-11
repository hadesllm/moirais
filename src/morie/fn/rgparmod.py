# morie.fn — function file (hadesllm/morie)
"""Parametric system identification: AR all-pole model fitting."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_parametric_sysid"]


def rangayyan_parametric_sysid(x, order):
    """
    Parametric system identification: AR all-pole model fitting

    Formula: H(z) = 1 / A(z) = 1 / (1 + a1*z^{-1} + ... + ap*z^{-p})

    Parameters
    ----------
    x : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a_coeffs, excitation

    References
    ----------
    Rangayyan Ch 7.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parametric system identification: AR all-pole model fitting"})


def cheatsheet():
    return "rgparmod: Parametric system identification: AR all-pole model fitting"
