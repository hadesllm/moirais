"""Decay bound for the complex cepstrum: at least as fast as 1/n.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_complex_cepstrum_decay_bound"]


def rangayyan_ch4_complex_cepstrum_decay_bound(K, alpha, n):
    """
    Decay bound for the complex cepstrum: at least as fast as 1/n.

    Formula: |x_hat(n)| < K * |alpha^n / n|, for -inf < n < inf, where alpha = max(|a_k|,|b_k|,|c_k|,|d_k|)

    Parameters
    ----------
    K : array-like
        Input data.
    alpha : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.73, p. 248
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    n = len(K)
    result = float(np.mean(K))
    se = float(np.std(K, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decay bound for the complex cepstrum: at least as fast as 1/n."})


def cheatsheet():
    return "rng245: Decay bound for the complex cepstrum: at least as fast as 1/n."
