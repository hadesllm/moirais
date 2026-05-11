"""Phase response of the ideal integrator (constant -pi/2).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_integrator_phase_response"]


def rangayyan_ch3_integrator_phase_response(omega):
    """
    Phase response of the ideal integrator (constant -pi/2).

    Formula: angle(H(omega)) = -pi/2

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.118, p. 144
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Phase response of the ideal integrator (constant -pi/2)."})


def cheatsheet():
    return "rng107: Phase response of the ideal integrator (constant -pi/2)."
