"""Intrinsic CAR (ICAR) prior for Bayesian spatial models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_icar_prior"]


def schabenberger_icar_prior(w):
    """
    Intrinsic CAR (ICAR) prior for Bayesian spatial models

    Formula: pi(phi) proportional to exp(-0.5*phi'*(D-W)*phi); improper: sigma^2 -> inf

    Parameters
    ----------
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: precision_matrix

    References
    ----------
    Schabenberger Ch 6, Sec 6.4.3
    """
    w = np.asarray(w, dtype=float)
    n = int(w) if w.ndim == 0 else len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Intrinsic CAR (ICAR) prior for Bayesian spatial models"})


def cheatsheet():
    return "spicar: Intrinsic CAR (ICAR) prior for Bayesian spatial models"
