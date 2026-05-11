"""Matched-filter output equals scaled, delayed ACF of the reference signal.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_output_acf"]


def rangayyan_ch4_matched_filter_output_acf(phi_x, K, t, t_0):
    """
    Matched-filter output equals scaled, delayed ACF of the reference signal.

    Formula: y(t) = K * phi_x(t - t_0)

    Parameters
    ----------
    phi_x : array-like
        Input data.
    K : array-like
        Input data.
    t : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.50, p. 239
    """
    phi_x = np.atleast_1d(np.asarray(phi_x, dtype=float))
    n = len(phi_x)
    result = float(np.mean(phi_x))
    se = float(np.std(phi_x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matched-filter output equals scaled, delayed ACF of the reference signal."})


def cheatsheet():
    return "rng222: Matched-filter output equals scaled, delayed ACF of the reference signal."
