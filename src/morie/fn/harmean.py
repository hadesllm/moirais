# morie.fn -- function file (hadesllm/morie)
"""Harmonic mean with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import hmean


def harmean(x: Union[Sequence[float], np.ndarray]):
    """Harmonic mean = n / Σ (1/x_i)."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("empty input.")
    if np.any(a <= 0):
        raise ValueError("all values must be positive.")
    h = float(hmean(a))
    arith = float(a.mean())
    return RichResult(
        title="Harmonic mean",
        summary_lines=[
            ("Harmonic mean", h),
            ("Arithmetic mean (for context)", arith),
            ("AM/HM ratio (>=1)", arith / h),
            ("n", int(a.size)),
        ],
        interpretation=("HM <= GM <= AM always. Use HM for averaging rates "
                        "(e.g., mph over equal distances)."),
        payload={"value": h, "statistic": h, "arithmetic_mean": arith},
    )
