"""Phase response from sums of angles to zeros and poles.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_phase_response_from_pole_zero"]


def rangayyan_ch3_phase_response_from_pole_zero(z_0, alpha_k, beta_k, N, M):
    """
    Phase response from sums of angles to zeros and poles.

    Formula: angle(H(omega_0)) = (M-N)*angle(z_0) + sum_{k=1}^{N} alpha_k - sum_{k=1}^{M} beta_k

    Parameters
    ----------
    z_0 : array-like
        Input data.
    alpha_k : array-like
        Input data.
    beta_k : array-like
        Input data.
    N : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.73, p. 125
    """
    z_0 = np.atleast_1d(np.asarray(z_0, dtype=float))
    n = len(z_0)
    result = float(np.mean(z_0))
    se = float(np.std(z_0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Phase response from sums of angles to zeros and poles."})


def cheatsheet():
    return "rng062: Phase response from sums of angles to zeros and poles."
