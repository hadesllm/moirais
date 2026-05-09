"""Phase response of the three-point central-difference operator.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_three_point_central_diff_phase"]


def rangayyan_ch3_three_point_central_diff_phase(omega):
    """
    Phase response of the three-point central-difference operator.

    Formula: angle(H(omega)) = pi/2 - omega

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
    Rangayyan (2024), Ch 3, Eq 3.131, p. 148
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Phase response of the three-point central-difference operator."})


def cheatsheet():
    return "rng119: Phase response of the three-point central-difference operator."
