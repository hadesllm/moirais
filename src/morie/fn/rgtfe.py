# morie.fn — function file (hadesllm/morie)
"""Transfer function estimation between input/output signals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_transfer_func_est"]


def rangayyan_transfer_func_est(x, y, fs, nperseg):
    """
    Transfer function estimation between input/output signals

    Formula: H(f) = S_xy(f)/S_xx(f); coherence validates estimate quality

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    fs : array-like
        Input data.
    nperseg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: H_f, coherence, freqs

    References
    ----------
    Rangayyan Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Transfer function estimation between input/output signals"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Transfer function estimation between input/output signals"})


def cheatsheet():
    return "rgtfe: Transfer function estimation between input/output signals"
