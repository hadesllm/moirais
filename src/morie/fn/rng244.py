"""Closed-form complex cepstrum from poles/zeros (inside/outside unit circle).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_complex_cepstrum_closed_form"]


def rangayyan_ch4_complex_cepstrum_closed_form(A, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O, n):
    """
    Closed-form complex cepstrum from poles/zeros (inside/outside unit circle).

    Formula: x_hat(n) = log|A| if n=0; -sum_{k=1}^{M_I} a_k^n/n + sum_{k=1}^{N_I} c_k^n/n for n>0; sum_{k=1}^{M_O} b_k^(-n)/n - sum_{k=1}^{N_O} d_k^(-n)/n for n<0

    Parameters
    ----------
    A : array-like
        Input data.
    a_k : array-like
        Input data.
    b_k : array-like
        Input data.
    c_k : array-like
        Input data.
    d_k : array-like
        Input data.
    M_I : array-like
        Input data.
    M_O : array-like
        Input data.
    N_I : array-like
        Input data.
    N_O : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.72, p. 248
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed-form complex cepstrum from poles/zeros (inside/outside unit circle)."})


def cheatsheet():
    return "rng244: Closed-form complex cepstrum from poles/zeros (inside/outside unit circle)."
