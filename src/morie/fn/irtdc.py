# morie.fn — function file (hadesllm/morie)
"""IRT discrimination summary statistics."""

from __future__ import annotations

from ._containers import DescriptiveResult


def irt_discrimination_summary(alpha) -> DescriptiveResult:
    """Summary statistics for discrimination parameters.

    .. epigraph:: "Shame! Shame!" -- Septa, Game of Thrones
    """
    import numpy as np

    a = np.asarray(alpha, dtype=float).ravel()
    return DescriptiveResult(
        name="irt_discrimination_summary",
        value=float(np.mean(a)),
        extra={
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
            "n_items": len(a),
        },
    )


irtdc = irt_discrimination_summary


def cheatsheet() -> str:
    return "irt_discrimination_summary({}) -> IRT discrimination summary statistics."
