"""Mean-squared (MS) value of a random process as the second-order moment of its PDF.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_mean_squared_value"]


def rangayyan_ch3_mean_squared_value(eta, p_eta):
    """
    Mean-squared (MS) value of a random process as the second-order moment of its PDF.

    Formula: E[eta^2] = integral_{-inf}^{inf} eta^2 * p_eta(eta) d(eta)

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
    Rangayyan (2024), Ch 3, Eq 3.2, p. 94
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    n = len(eta)
    result = float(np.mean(eta))
    se = float(np.std(eta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean-squared (MS) value of a random process as the second-order moment of its PDF."})


def cheatsheet():
    return "rng002: Mean-squared (MS) value of a random process as the second-order moment of its PDF."
