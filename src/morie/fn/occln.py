# morie.fn -- function file (hadesllm/morie)
"""OC cutting line equation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def oc_cutting_line(normal, cutpoint) -> DescriptiveResult:
    """Compute the cutting line from normal vector and cutpoint.

    .. epigraph:: "It's all good, man." -- Saul, Better Call Saul
    """
    import numpy as np

    n = np.asarray(normal, dtype=float)
    c = float(cutpoint)
    if len(n) >= 2 and abs(n[1]) > 1e-12:
        slope = -n[0] / n[1]
        intercept = c / n[1]
    else:
        slope = float("inf")
        intercept = c / n[0] if abs(n[0]) > 1e-12 else 0.0
    return DescriptiveResult(
        name="oc_cutting_line",
        value=slope,
        extra={
            "slope": slope,
            "intercept": intercept,
            "normal": n.tolist(),
            "cutpoint": c,
        },
    )


occln = oc_cutting_line


def cheatsheet() -> str:
    return "oc_cutting_line({}) -> OC cutting line equation."
