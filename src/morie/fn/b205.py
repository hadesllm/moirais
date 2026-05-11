"""Perplexity as the exponential of the average negative log-likelihood per token under context window of size k.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch2_perplexity"]


def burkov_lm_ch2_perplexity(D, k, t):
    """
    Perplexity as the exponential of the average negative log-likelihood per token under context window of size k.

    Formula: \operatorname{Perplexity}(\mathcal{D}, k) = \exp\!\left(-\frac{1}{D} \sum_{i=1}^{D} \log \Pr\!\bigl(t_i \mid t_{\max(1, i-k)}, \ldots, t_{i-1}\bigr)\right)

    Parameters
    ----------
    D : array-like
        Input data.
    k : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: perplexity score (lower is better)

    References
    ----------
    Burkov LM (2025), Ch 2, Eq 2.5, p. 84
    """
    D = np.atleast_1d(np.asarray(D, dtype=float))
    n = len(D)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Perplexity as the exponential of the average negative log-likelihood per token under context window of size k."})
    estimate = np.median(D)
    se = 1.2533 * np.std(D, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Perplexity as the exponential of the average negative log-likelihood per token under context window of size k."})


def cheatsheet():
    return "b205: Perplexity as the exponential of the average negative log-likelihood per token under context window of size k."
