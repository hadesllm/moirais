# moirais.fn — function file (hadesllm/moirais)
"""SAR Cochrane-Orcutt-style spatial filter."""

import numpy as np

from ._containers import SpatialResult


def sarfilt(y, W, rho=0.3):
    """SAR Cochrane-Orcutt-style spatial filter.

    Category: SAR

    Parameters
    ----------
    y, W, rho=0.3 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="sarfilt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sarfilt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sarfilt_fn = sarfilt


def cheatsheet() -> str:
    return "sarfilt({}) -> SAR Cochrane-Orcutt-style spatial filter."
