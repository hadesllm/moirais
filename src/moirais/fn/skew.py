# moirais.fn — function file (hadesllm/moirais)
"""Sample skewness with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import skew as _scipy_skew


def skew(x: Union[Sequence[float], np.ndarray], bias: bool = False):
    """Sample skewness (Fisher-Pearson g_1)."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    if a.size < 3:
        raise ValueError(f"need at least 3 observations, got {a.size}.")
    g1 = float(_scipy_skew(a, bias=bias))
    if abs(g1) < 0.5: shape = "approximately symmetric"
    elif abs(g1) < 1: shape = "moderately skewed"
    else: shape = "heavily skewed"
    direction = "right" if g1 > 0 else "left"
    return RichResult(
        title="Sample skewness",
        summary_lines=[
            ("Skewness g_1", g1),
            ("Shape", shape),
            ("Direction", direction if abs(g1) > 0.05 else "≈ 0"),
            ("n", int(a.size)),
            ("Bias correction", "off (raw m_3 / m_2^1.5)" if bias else "Fisher-Pearson"),
        ],
        interpretation=(f"g_1 = {g1:+.3f}; magnitude {shape}, "
                        f"{'right (positive) tail longer' if g1 > 0 else 'left (negative) tail longer' if g1 < 0 else 'symmetric'}. "
                        "Normal = 0."),
        payload={"value": g1, "statistic": g1, "shape": shape},
    )
