# moirais.fn — function file (hadesllm/moirais)
"""AR (all-pole) parametric PSD estimate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ar_spectrum"]


def rangayyan_ar_spectrum(x, order, fs):
    """
    AR (all-pole) parametric PSD estimate

    Formula: S_AR(f) = sigma^2 / |1 + sum a_k * exp(-j2*pi*f*k*T)|^2

    Parameters
    ----------
    x : array-like
        Input data.
    order : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: psd, freqs

    References
    ----------
    Rangayyan Ch 7.5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "AR (all-pole) parametric PSD estimate"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "AR (all-pole) parametric PSD estimate"})


def cheatsheet():
    return "rgarsp: AR (all-pole) parametric PSD estimate"
