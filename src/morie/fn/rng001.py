"""Mean of a random process as the first-order moment of its PDF.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_mean_continuous"]


def rangayyan_ch3_mean_continuous(eta, p_eta):
    """
    Mean of a random process as the first-order moment of its PDF.

    Formula: mu_eta = E[eta] = integral_{-inf}^{inf} eta * p_eta(eta) d(eta)

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
    Rangayyan (2024), Ch 3, Eq 3.1, p. 94
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    n = len(eta)
    result = float(np.mean(eta))
    se = float(np.std(eta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean of a random process as the first-order moment of its PDF."})


def cheatsheet():
    return "rng001: Mean of a random process as the first-order moment of its PDF."
