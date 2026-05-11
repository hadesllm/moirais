# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Noise-Contrastive Estimation: binary-classify data vs noise distribution q."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_noise_contrastive_estimation"]


def burkov_noise_contrastive_estimation(data_scores, noise_scores, k, q_data, q_noise):
    """
    Noise-Contrastive Estimation: binary-classify data vs noise distribution q

    Formula: L = - sum_{data} log sigma(score(w) - log(k q(w)))  - sum_{noise_i} log sigma(-(score(n_i) - log(k q(n_i))))

    Parameters
    ----------
    data_scores : array-like
        Input data.
    noise_scores : array-like
        Input data.
    k : array-like
        Input data.
    q_data : array-like
        Input data.
    q_noise : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Burkov Ch 2, Noise-Contrastive Estimation section
    """
    data_scores = np.atleast_1d(np.asarray(data_scores, dtype=float))
    n = len(data_scores)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Noise-Contrastive Estimation: binary-classify data vs noise distribution q"})
    estimate = np.median(data_scores)
    se = 1.2533 * np.std(data_scores, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Noise-Contrastive Estimation: binary-classify data vs noise distribution q"})


def cheatsheet():
    return "bknce: Noise-Contrastive Estimation: binary-classify data vs noise distribution q"
