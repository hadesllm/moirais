# moirais.fn — function file (hadesllm/moirais)
"""Median Absolute Deviation with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def mad(x: Union[Sequence[float], np.ndarray], scale: str = "normal"):
    """Median Absolute Deviation."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        raise ValueError("need at least 2 observations.")
    median = float(np.median(a))
    raw = float(np.median(np.abs(a - median)))
    if scale == "normal":
        v = 1.4826 * raw
        scale_label = "normal (×1.4826)"
    elif scale == "raw":
        v = raw
        scale_label = "raw (no rescale)"
    else:
        v = float(scale) * raw
        scale_label = f"custom (×{scale})"
    sample_sd = float(a.std(ddof=1))
    return RichResult(
        title="Median Absolute Deviation",
        summary_lines=[
            ("MAD (scaled)", v),
            ("Raw MAD", raw),
            ("Sample SD (for context)", sample_sd),
            ("MAD/SD ratio", v / sample_sd if sample_sd > 0 else float("nan")),
            ("Scale", scale_label),
            ("Median", median),
            ("n", int(a.size)),
        ],
        interpretation=("MAD is robust to outliers; under Normality, "
                        "1.4826*MAD ≈ SD. If MAD/SD << 1, the sample SD "
                        "is being inflated by outliers."),
        payload={"value": v, "statistic": v, "raw": raw, "median": median},
    )
