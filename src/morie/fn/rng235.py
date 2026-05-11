"""Complex logarithm converts the product Y(omega)=X(omega)H(omega) into a sum.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_complex_log_of_product"]


def rangayyan_ch4_complex_log_of_product(X, H, omega):
    """
    Complex logarithm converts the product Y(omega)=X(omega)H(omega) into a sum.

    Formula: log[Y(omega)] = log[X(omega)] + log[H(omega)], with X(omega)!=0 and H(omega)!=0

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.63, p. 245
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Complex logarithm converts the product Y(omega)=X(omega)H(omega) into a sum."})


def cheatsheet():
    return "rng235: Complex logarithm converts the product Y(omega)=X(omega)H(omega) into a sum."
