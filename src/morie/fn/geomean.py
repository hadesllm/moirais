# morie.fn -- function file (hadesllm/morie)
"""Geometric mean with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import gmean


def geomean(x: Union[Sequence[float], np.ndarray]):
    """Geometric mean = (Π_i x_i)^(1/n)."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("empty input.")
    if np.any(a <= 0):
        raise ValueError("all values must be positive.")
    g = float(gmean(a))
    arith = float(a.mean())
    return RichResult(
        title="Geometric mean",
        summary_lines=[
            ("Geometric mean", g),
            ("Arithmetic mean (for context)", arith),
            ("AM/GM ratio (>=1)", arith / g),
            ("n", int(a.size)),
        ],
        interpretation=("AM >= GM always. Use GM for ratios, growth rates, "
                        "or log-Normal data. AM/GM gap reflects variance."),
        payload={"value": g, "statistic": g, "arithmetic_mean": arith},
    )
