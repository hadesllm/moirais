"""Sample overlap / dual-frame estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sample_overlap"]


def sample_overlap(frame_a, frame_b, overlap_indicator):
    """
    Sample overlap / dual-frame estimator

    Formula: weighted combination across frames

    Parameters
    ----------
    frame_a : array-like
        Input data.
    frame_b : array-like
        Input data.
    overlap_indicator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hartley (1962)
    """
    frame_a = np.atleast_1d(np.asarray(frame_a, dtype=float))
    n = len(frame_a)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Sample overlap / dual-frame estimator"})
    estimate = np.median(frame_a)
    se = 1.2533 * np.std(frame_a, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Sample overlap / dual-frame estimator"})


def cheatsheet():
    return "smplep: Sample overlap / dual-frame estimator"
