# moirais.fn — function file (hadesllm/moirais)
"""Trimmed mean with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import trim_mean


def trmean(x: Union[Sequence[float], np.ndarray], trim: float = 0.1):
    """Symmetric trimmed mean."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    if not 0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim}.")
    if a.size < 2:
        raise ValueError("need at least 2 observations.")
    tm = float(trim_mean(a, trim))
    arith = float(a.mean())
    n_trimmed = int(2 * np.floor(a.size * trim))
    return RichResult(
        title=f"Trimmed mean (trim={trim*100:.0f}% per tail)",
        summary_lines=[
            ("Trimmed mean", tm),
            ("Plain arithmetic mean", arith),
            ("Difference (trimmed - plain)", tm - arith),
            ("n", int(a.size)),
            ("Observations trimmed", f"{n_trimmed} of {a.size}"),
        ],
        interpretation=("Trimmed mean is robust to outliers; if it differs "
                        "substantially from the plain mean, your data has "
                        "tail influence. Wilcox 2017 ch.5 recommends 20% trim."),
        payload={"value": tm, "statistic": tm, "trim": trim,
                 "arithmetic_mean": arith, "n_trimmed": n_trimmed},
    )
