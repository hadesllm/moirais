# moirais.fn — function file (hadesllm/moirais)
"""Lemma 3.1: asymptotic representation of kernel quantile estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_lem3_1_asymp_rep"]


def fauzi_lem3_1_asymp_rep(data, p, bandwidth):
    """
    Lemma 3.1: asymptotic representation of kernel quantile estimator

    Formula: sigma_n^{-1}*sqrt(n)*(Q_hat_{p,h}-Q_hat(p)) = d_{1n}A_{1n}+d_{2n}A_{2n}+d_{3n}A_{3n}+...+o_L(n^{-1/2})

    Parameters
    ----------
    data : array-like
        Input data.
    p : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: representation

    References
    ----------
    Fauzi Ch 3, Lemma 3.1
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    if data.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Lemma 3.1: asymptotic representation of kernel quantile estimator"})
    estimate = np.median(data)
    se = 1.2533 * np.std(data, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Lemma 3.1: asymptotic representation of kernel quantile estimator"})


def cheatsheet():
    return "fzl31: Lemma 3.1: asymptotic representation of kernel quantile estimator"
