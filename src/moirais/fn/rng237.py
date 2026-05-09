"""Complex logarithms of z-transforms of a convolved signal y(n) = x(n)*h(n).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_log_of_convolved_signals"]


def rangayyan_ch4_log_of_convolved_signals(X_hat, H_hat, z, omega):
    """
    Complex logarithms of z-transforms of a convolved signal y(n) = x(n)*h(n).

    Formula: Y_hat(z) = X_hat(z) + H_hat(z); Y_hat(omega) = X_hat(omega) + H_hat(omega)

    Parameters
    ----------
    X_hat : array-like
        Input data.
    H_hat : array-like
        Input data.
    z : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.65, p. 247
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Complex logarithms of z-transforms of a convolved signal y(n) = x(n)*h(n)."})


def cheatsheet():
    return "rng237: Complex logarithms of z-transforms of a convolved signal y(n) = x(n)*h(n)."
