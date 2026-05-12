r"""Maximum-likelihood estimate of a trigram language-model probability using n-gram counts from a corpus.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch2_trigram_count"]


def burkov_lm_ch2_trigram_count(t_i, t_im1, t_im2):
    r"""
    Maximum-likelihood estimate of a trigram language-model probability using n-gram counts from a corpus.

    Formula: \Pr(t_i \mid t_{i-2}, t_{i-1}) = \frac{C(t_{i-2}, t_{i-1}, t_i)}{C(t_{i-2}, t_{i-1})}

    Parameters
    ----------
    t_i : array-like
        Input data.
    t_im1 : array-like
        Input data.
    t_im2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: trigram conditional probability

    References
    ----------
    Burkov LM (2025), Ch 2, Eq 2.4, p. 77
    """
    t_i = np.atleast_1d(np.asarray(t_i, dtype=float))
    n = len(t_i)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Maximum-likelihood estimate of a trigram language-model probability using n-gram counts from a corpus."})
    estimate = np.median(t_i)
    se = 1.2533 * np.std(t_i, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Maximum-likelihood estimate of a trigram language-model probability using n-gram counts from a corpus."})


def cheatsheet():
    return "b204: Maximum-likelihood estimate of a trigram language-model probability using n-gram counts from a corpus."
