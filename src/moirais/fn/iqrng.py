# moirais.fn — function file (hadesllm/moirais)
"""Interquartile range with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import iqr as _scipy_iqr


def iqrng(x: Union[Sequence[float], np.ndarray], method: str = "linear"):
    """Interquartile range = Q3 - Q1."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    if a.size < 4:
        raise ValueError(f"need at least 4 observations for stable IQR, got {a.size}.")
    q1 = float(np.quantile(a, 0.25, method=method))
    q3 = float(np.quantile(a, 0.75, method=method))
    iqr_val = float(_scipy_iqr(a, interpolation=method))
    median = float(np.median(a))
    return RichResult(
        title="Interquartile range (IQR)",
        summary_lines=[
            ("IQR (Q3 - Q1)", iqr_val),
            ("Q1 (25th percentile)", q1),
            ("Median (Q2)", median),
            ("Q3 (75th percentile)", q3),
            ("Lower fence (Q1 - 1.5*IQR)", q1 - 1.5 * iqr_val),
            ("Upper fence (Q3 + 1.5*IQR)", q3 + 1.5 * iqr_val),
            ("n", int(a.size)),
            ("Method", method),
        ],
        interpretation=("Robust spread; values outside [Q1 - 1.5*IQR, "
                        "Q3 + 1.5*IQR] are flagged as outliers in box-plot "
                        "convention."),
        payload={"value": iqr_val, "statistic": iqr_val,
                 "q1": q1, "q3": q3, "median": median},
    )
