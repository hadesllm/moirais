# morie.fn — function file (hadesllm/morie)
"""Sample kurtosis with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import kurtosis as _scipy_kurtosis


def kurt(x: Union[Sequence[float], np.ndarray],
         fisher: bool = True, bias: bool = False):
    """Sample kurtosis (Fisher excess g_2)."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    if a.size < 4:
        raise ValueError(f"need at least 4 observations, got {a.size}.")
    g2 = float(_scipy_kurtosis(a, fisher=fisher, bias=bias))
    if fisher:
        if abs(g2) < 0.5: shape = "mesokurtic (~Normal)"
        elif g2 > 0.5: shape = "leptokurtic (heavy tails)"
        else: shape = "platykurtic (thin tails)"
    else:
        # Pearson kurtosis: 3 = Normal
        if abs(g2 - 3) < 0.5: shape = "mesokurtic (~Normal)"
        elif g2 > 3.5: shape = "leptokurtic (heavy tails)"
        else: shape = "platykurtic (thin tails)"
    return RichResult(
        title="Sample kurtosis",
        summary_lines=[
            ("Kurtosis g_2", g2),
            ("Shape", shape),
            ("Convention", "Fisher excess (Normal = 0)" if fisher else "Pearson (Normal = 3)"),
            ("n", int(a.size)),
        ],
        interpretation=(f"g_2 = {g2:+.3f}; {shape}. "
                        + ("Excess kurtosis: Normal = 0." if fisher else "Pearson: Normal = 3.")),
        payload={"value": g2, "statistic": g2, "shape": shape},
    )
