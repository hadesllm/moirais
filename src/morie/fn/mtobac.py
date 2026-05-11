# morie.fn — function file (hadesllm/morie)
"""BAC distribution in impaired driving."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def mto_bac_analysis(
    bac_levels: np.ndarray | list[float],
    *,
    legal_limit: float = 0.08,
) -> DescriptiveResult:
    """Analyse BAC distribution in impaired driving cases.

    Parameters
    ----------
    bac_levels : array-like
        Blood alcohol concentration values.
    legal_limit : float
        Legal BAC limit (default 0.08).

    Returns
    -------
    DescriptiveResult
    """
    b = np.asarray(bac_levels, dtype=float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        raise ValueError("No valid BAC levels")
    over_limit = float(np.mean(b > legal_limit))
    return DescriptiveResult(
        name="bac_analysis",
        value=float(np.mean(b)),
        extra={
            "mean": float(np.mean(b)),
            "median": float(np.median(b)),
            "std": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
            "pct_over_limit": over_limit,
            "legal_limit": legal_limit,
            "n": len(b),
        },
    )


mtobac = mto_bac_analysis


def cheatsheet() -> str:
    return "mto_bac_analysis({}) -> BAC distribution in impaired driving."
