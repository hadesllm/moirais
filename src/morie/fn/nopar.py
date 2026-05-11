# morie.fn — function file (hadesllm/morie)
"""NOMINATE bill parameter summary."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nominate_parameters(votes, X, nv, mid) -> DescriptiveResult:
    """Summarise NOMINATE bill parameters (normal vectors, midpoints).

    .. epigraph:: "No half measures." -- Mike, Breaking Bad
    """
    import numpy as np

    nv = np.asarray(nv, dtype=float)
    mid = np.asarray(mid, dtype=float)
    return DescriptiveResult(
        name="nominate_parameters",
        value=float(np.mean(mid)),
        extra={
            "n_bills": len(mid),
            "nv_mean": float(np.mean(nv)),
            "nv_std": float(np.std(nv)),
            "mid_mean": float(np.mean(mid)),
            "mid_std": float(np.std(mid)),
            "mid_range": [float(np.min(mid)), float(np.max(mid))],
        },
    )


nopar = nominate_parameters


def cheatsheet() -> str:
    return "nominate_parameters({}) -> NOMINATE bill parameter summary."
