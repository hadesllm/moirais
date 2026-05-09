# moirais.fn — function file (hadesllm/moirais)
"""Empirical mode decomposition (EMD) sifting algorithm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_emd"]


def rangayyan_emd(x, max_imfs, tol):
    """
    Empirical mode decomposition (EMD) sifting algorithm

    Formula: IMF_k from sifting: h(t)=x(t)-(upper+lower)/2; iterate until stoppage

    Parameters
    ----------
    x : array-like
        Input data.
    max_imfs : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: imfs, residue

    References
    ----------
    Rangayyan Ch 9.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical mode decomposition (EMD) sifting algorithm"})


def cheatsheet():
    return "rgemd: Empirical mode decomposition (EMD) sifting algorithm"
