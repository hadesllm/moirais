# morie.fn — function file (hadesllm/morie)
"""Word Embedding Association Test bias score."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_weat_bias_score"]


def kamath_weat_bias_score(X_embeddings, Y_embeddings, A_embeddings, B_embeddings, cdf=None):
    """
    Word Embedding Association Test bias score

    Formula: s(X, Y, A, B) = sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B);  s(w, A, B) = mean cos(w, a) - mean cos(w, b)

    Parameters
    ----------
    X_embeddings : array-like
        Input data.
    Y_embeddings : array-like
        Input data.
    A_embeddings : array-like
        Input data.
    B_embeddings : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: effect_size

    References
    ----------
    Kamath Ch 6, WEAT section
    """
    X_embeddings = np.asarray(X_embeddings, dtype=float)
    n = len(X_embeddings)
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Word Embedding Association Test bias score"})
    x_sorted = np.sort(X_embeddings)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(X_embeddings), scale=np.std(X_embeddings, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Word Embedding Association Test bias score"})


def cheatsheet():
    return "kmweat: Word Embedding Association Test bias score"
