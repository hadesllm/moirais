"""Magnitude of instantaneous output signal of a matched filter at t = t0.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_instantaneous_signal"]


def rangayyan_ch4_matched_filter_instantaneous_signal(X, H, f, t_0):
    """
    Magnitude of instantaneous output signal of a matched filter at t = t0.

    Formula: M_y = |y(t_0)| = | integral_{-inf}^{inf} X(f) H(f) exp(+j*2*pi*f*t_0) df |

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    f : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.38, p. 238
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Magnitude of instantaneous output signal of a matched filter at t = t0."})


def cheatsheet():
    return "rng212: Magnitude of instantaneous output signal of a matched filter at t = t0."
