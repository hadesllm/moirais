# morie.fn — function file (hadesllm/morie)
"""Sample covariance with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def covar(x: Union[Sequence[float], np.ndarray],
          y: Union[Sequence[float], np.ndarray]):
    """Unbiased sample covariance with verbose result."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float); b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}.")
    if a.size < 2:
        raise ValueError("need at least 2 observations.")
    cov_val = float(np.cov(a, b, ddof=1)[0, 1])
    sd_a = float(a.std(ddof=1)); sd_b = float(b.std(ddof=1))
    corr = cov_val / (sd_a * sd_b) if sd_a > 0 and sd_b > 0 else float("nan")
    return RichResult(
        title="Sample covariance",
        summary_lines=[
            ("Cov(x, y)", cov_val),
            ("Pearson r (for context)", corr),
            ("SD(x)", sd_a), ("SD(y)", sd_b),
            ("n pairs", int(a.size)),
        ],
        interpretation=("Covariance is scale-dependent; Pearson r normalized "
                        f"= {corr:+.4f}. Use `kentau` or `spearm` for non-parametric "
                        "alternatives."),
        payload={"value": cov_val, "statistic": cov_val,
                 "correlation": corr, "sd_x": sd_a, "sd_y": sd_b},
    )
