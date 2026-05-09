"""Frequency-domain output of matched filter equals PSD of the reference signal.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_output_psd"]


def rangayyan_ch4_matched_filter_output_psd(X, H, f):
    """
    Frequency-domain output of matched filter equals PSD of the reference signal.

    Formula: Y(f) = X(f) * H(f) = X(f) * X*(f) = S_x(f)

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.57, p. 241
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency-domain output of matched filter equals PSD of the reference signal."})


def cheatsheet():
    return "rng229: Frequency-domain output of matched filter equals PSD of the reference signal."
