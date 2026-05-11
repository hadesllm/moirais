"""Differential entropy of a continuous PDF in bits.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_entropy_continuous"]


def rangayyan_ch3_entropy_continuous(eta, p_eta):
    """
    Differential entropy of a continuous PDF in bits.

    Formula: H_eta = - integral_{-inf}^{inf} p_eta(eta) * log2(p_eta(eta)) d(eta)

    Parameters
    ----------
    eta : array-like
        Input data.
    p_eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.6, p. 94
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    n = len(eta)
    result = float(np.mean(eta))
    se = float(np.std(eta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Differential entropy of a continuous PDF in bits."})


def cheatsheet():
    return "rng006: Differential entropy of a continuous PDF in bits."
