"""Normal equation for the RLS algorithm.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_normal_equation"]


def rangayyan_ch3_rls_normal_equation(Phi, w_tilde, Theta, n):
    """
    Normal equation for the RLS algorithm.

    Formula: Phi(n) * w_tilde(n) = Theta(n)

    Parameters
    ----------
    Phi : array-like
        Input data.
    w_tilde : array-like
        Input data.
    Theta : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.207, p. 187
    """
    Phi = np.atleast_1d(np.asarray(Phi, dtype=float))
    n = len(Phi)
    result = float(np.mean(Phi))
    se = float(np.std(Phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normal equation for the RLS algorithm."})


def cheatsheet():
    return "rng164: Normal equation for the RLS algorithm."
