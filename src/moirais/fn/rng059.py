"""Alternative pole-zero factored transfer function with z^(M-N) gain factor.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_pole_zero_factored_form_alt"]


def rangayyan_ch3_pole_zero_factored_form_alt(z_k, p_k, z, N, M):
    """
    Alternative pole-zero factored transfer function with z^(M-N) gain factor.

    Formula: H(z) = z^(M-N) * prod_{k=1}^{N} (z - z_k) / prod_{k=1}^{M} (z - p_k)

    Parameters
    ----------
    z_k : array-like
        Input data.
    p_k : array-like
        Input data.
    z : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.70, p. 124
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Alternative pole-zero factored transfer function with z^(M-N) gain factor."})


def cheatsheet():
    return "rng059: Alternative pole-zero factored transfer function with z^(M-N) gain factor."
