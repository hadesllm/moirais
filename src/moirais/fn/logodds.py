# moirais.fn — function file (hadesllm/moirais)
"""Log-odds (logit) transform with R-style verbose result."""

import math
from typing import Sequence, Union
import numpy as np


def logodds(p: Union[float, Sequence[float], np.ndarray]):
    """Log-odds (logit): ln(p / (1 - p))."""
    if isinstance(p, (int, float)):
        from ._richresult import RichResult
        if not (0 < p < 1):
            raise ValueError(f"p must be in (0, 1), got {p}.")
        l = math.log(p / (1.0 - p))
        return RichResult(
            title="Log-odds (logit) transform",
            summary_lines=[
                ("logit(p)", l),
                ("p", p),
                ("Odds (p / (1-p))", p / (1 - p)),
                ("Inverse: 1 / (1 + e^-logit)", 1 / (1 + math.exp(-l))),
            ],
            interpretation=("Logit maps (0, 1) → ℝ; the link function in "
                            "logistic regression. Inverse via `invlgt`."),
            payload={"value": l, "statistic": l, "odds": p / (1 - p)},
        )
    # Array input — return raw array (skip RichResult overhead)
    a = np.asarray(p, dtype=float)
    if np.any((a <= 0) | (a >= 1)):
        raise ValueError("all elements of p must be in (0, 1).")
    return np.log(a / (1.0 - a))
