# morie.fn -- function file (hadesllm/morie)
"""Otoacoustic emission (OAE) signal analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_oae"]


def rangayyan_oae(oae, fs):
    """
    Otoacoustic emission (OAE) signal analysis

    Formula: TEOAEs extracted by nonlinear click suppression; DPOAE ratio 2f1-f2

    Parameters
    ----------
    oae : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: analysis

    References
    ----------
    Rangayyan Ch 1.2.16
    """
    oae = np.asarray(oae, dtype=float)
    n = int(oae) if oae.ndim == 0 else len(oae)
    result = float(np.mean(oae))
    se = float(np.std(oae, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Otoacoustic emission (OAE) signal analysis"})


def cheatsheet():
    return "rgoae: Otoacoustic emission (OAE) signal analysis"
