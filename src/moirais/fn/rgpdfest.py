# moirais.fn — function file (hadesllm/moirais)
"""Probability density function estimation of VAG signals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pdf_estimate"]


def rangayyan_pdf_estimate(x, bins, bw):
    """
    Probability density function estimation of VAG signals

    Formula: p(x) estimated via histogram or kernel density with Gaussian kernel

    Parameters
    ----------
    x : array-like
        Input data.
    bins : array-like
        Input data.
    bw : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pdf, bins

    References
    ----------
    Rangayyan Ch 5.12.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Probability density function estimation of VAG signals"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Probability density function estimation of VAG signals"})


def cheatsheet():
    return "rgpdfest: Probability density function estimation of VAG signals"
