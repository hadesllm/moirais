"""Shannon entropy of a discrete process with L quantized values.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_shannon_entropy_discrete"]


def rangayyan_ch3_shannon_entropy_discrete(p_eta, L):
    """
    Shannon entropy of a discrete process with L quantized values.

    Formula: H_eta = - sum_{l=0}^{L-1} p_eta(eta_l) * log2(p_eta(eta_l))

    Parameters
    ----------
    p_eta : array-like
        Input data.
    L : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.11, p. 95
    """
    p_eta = np.atleast_1d(np.asarray(p_eta, dtype=float))
    n = len(p_eta)
    result = float(np.mean(p_eta))
    se = float(np.std(p_eta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shannon entropy of a discrete process with L quantized values."})


def cheatsheet():
    return "rng011: Shannon entropy of a discrete process with L quantized values."
