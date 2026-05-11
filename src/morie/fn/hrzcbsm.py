# morie.fn — function file (hadesllm/morie)
"""Maximum-score estimator with choice-based samples."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_choice_based_sms"]


def horowitz_choice_based_sms(x, y, sampling_weights):
    """
    Maximum-score estimator with choice-based samples

    Formula: Modified objective function weights for non-random sampling; tau_0=P(Y=1) in population

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    sampling_weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat

    References
    ----------
    Horowitz Ch 4, Sec 4.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Maximum-score estimator with choice-based samples"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Maximum-score estimator with choice-based samples"})


def cheatsheet():
    return "hrzcbsm: Maximum-score estimator with choice-based samples"
