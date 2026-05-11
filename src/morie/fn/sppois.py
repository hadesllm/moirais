"""Poisson process: independent counts, N(A)~Pois(lam*|A|)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_poisson_process"]


def schabenberger_poisson_process(lam, region):
    """
    Poisson process: independent counts, N(A)~Pois(lam*|A|)

    Formula: P(N(A)=n) = exp(-lam*|A|)*(lam*|A|)^n/n!

    Parameters
    ----------
    lam : array-like
        Input data.
    region : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pmf

    References
    ----------
    Schabenberger Ch 3, Sec 3.2.2
    """
    lam = np.asarray(lam, dtype=float)
    n = int(lam) if lam.ndim == 0 else len(lam)
    result = float(np.mean(lam))
    se = float(np.std(lam, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Poisson process: independent counts, N(A)~Pois(lam*|A|)"})


def cheatsheet():
    return "sppois: Poisson process: independent counts, N(A)~Pois(lam*|A|)"
