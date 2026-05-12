# morie.fn -- function file (hadesllm/morie)
"""SF-12 mental component summary (simplified)."""

import numpy as np

from ._containers import ESRes


def sf12_mental(
    items: list | np.ndarray,
) -> ESRes:
    """Simplified SF-12 Mental Component Summary (MCS-12).

    Uses unweighted sum of mental-health-related items scaled to 0-100.
    For production use, apply published scoring coefficients.

    Parameters
    ----------
    items : array-like
        Twelve SF-12 item responses.

    Returns
    -------
    ESRes

    References
    ----------
    Ware, J. E., Kosinski, M., & Keller, S. D. (1996).
    A 12-item Short-Form Health Survey. Medical Care, 34(3), 220-233.
    """
    a = np.asarray(items, dtype=float)
    if len(a) != 12:
        raise ValueError("SF-12 requires exactly 12 items")

    mental_items = a[[3, 4, 6, 7, 9, 10, 11]]
    raw = float(np.sum(mental_items))
    max_possible = 7 * np.max(a) if np.max(a) > 0 else 1
    scaled = raw / max_possible * 100 if max_possible > 0 else 0.0

    return ESRes(
        measure="SF-12_MCS",
        estimate=float(scaled),
        extra={"raw_sum": raw, "n_items": 12},
    )


mhsfr = sf12_mental


def cheatsheet() -> str:
    return "sf12_mental({}) -> SF-12 mental component summary (simplified)."
