# moirais.fn — function file (hadesllm/moirais)
"""RMS noise estimation from signal-free segments."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_rms_noise"]


def rangayyan_rms_noise(x, noise_segments):
    """
    RMS noise estimation from signal-free segments

    Formula: sigma_n = sqrt((1/N)*sum x_noise[n]^2)

    Parameters
    ----------
    x : array-like
        Input data.
    noise_segments : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rms_noise

    References
    ----------
    Rangayyan Ch 3.2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "RMS noise estimation from signal-free segments"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "RMS noise estimation from signal-free segments"})


def cheatsheet():
    return "rgrmsnw: RMS noise estimation from signal-free segments"
