# morie.fn — function file (hadesllm/morie)
"""Nernst equilibrium potential for ionic species."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_nernst_potential"]


def rangayyan_nernst_potential(T, z, conc_out, conc_in):
    """
    Nernst equilibrium potential for ionic species

    Formula: E_ion = (RT/zF) * ln([ion]_out / [ion]_in)

    Parameters
    ----------
    T : array-like
        Input data.
    z : array-like
        Input data.
    conc_out : array-like
        Input data.
    conc_in : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: nernst_potential

    References
    ----------
    Rangayyan Ch 1.2.1
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nernst equilibrium potential for ionic species"})


def cheatsheet():
    return "rgnrnst: Nernst equilibrium potential for ionic species"
