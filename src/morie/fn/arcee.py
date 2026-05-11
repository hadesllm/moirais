# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Arc length computation. 'Speed is my weapon.' -- Arcee"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def arc_length(
    x: np.ndarray,
    y: np.ndarray,
    *,
    z: np.ndarray | None = None,
) -> DescriptiveResult:
    """Compute the arc length of a parametric curve via trapezoidal integration.

    For 2-D: L = sum(sqrt(dx^2 + dy^2))
    For 3-D: L = sum(sqrt(dx^2 + dy^2 + dz^2))

    Parameters
    ----------
    x, y : array-like
        Coordinates of the curve.
    z : array-like, optional
        Third coordinate for 3-D curves.

    Returns
    -------
    DescriptiveResult
        With ``value`` = total arc length.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 points")

    dx = np.diff(x)
    dy = np.diff(y)
    ds_sq = dx**2 + dy**2

    dim = 2
    if z is not None:
        z = np.asarray(z, dtype=float).ravel()
        if len(z) != len(x):
            raise ValueError("z must have same length as x")
        dz = np.diff(z)
        ds_sq += dz**2
        dim = 3

    segments = np.sqrt(ds_sq)
    total = float(segments.sum())
    cumulative = np.concatenate([[0], np.cumsum(segments)])

    return DescriptiveResult(
        name="arc_length",
        value=total,
        extra={"n_points": len(x), "dim": dim, "cumulative": cumulative, "segments": segments},
    )


arcee = arc_length


def cheatsheet() -> str:
    return "arc_length({}) -> Arc length computation. 'Speed is my weapon.' -- Arcee"
