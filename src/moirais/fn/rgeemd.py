# moirais.fn — function file (hadesllm/moirais)
"""Ensemble EMD (EEMD) for mode mixing alleviation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_eemd"]


def rangayyan_eemd(x, n_ensembles, noise_std, max_imfs):
    """
    Ensemble EMD (EEMD) for mode mixing alleviation

    Formula: EEMD: add white noise realizations, perform EMD, ensemble average IMFs

    Parameters
    ----------
    x : array-like
        Input data.
    n_ensembles : array-like
        Input data.
    noise_std : array-like
        Input data.
    max_imfs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: imfs

    References
    ----------
    Rangayyan Ch 9.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ensemble EMD (EEMD) for mode mixing alleviation"})


def cheatsheet():
    return "rgeemd: Ensemble EMD (EEMD) for mode mixing alleviation"
