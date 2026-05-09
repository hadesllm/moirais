# moirais.fn — function file (hadesllm/moirais)
"""Preference circles for visualization. 'Instant Transmission.' -- Goku, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def preference_circles(ideal_point, radius, n_points=100):
    """Generate circle points centered on an ideal point.

    Parameters
    ----------
    ideal_point : array-like
        2D coordinates of ideal point [x, y].
    radius : float
        Circle radius (preference threshold).
    n_points : int
        Number of points on circle.

    Returns
    -------
    DescriptiveResult
        value = circle coordinates (n_points x 2).
    """
    import numpy as np

    center = np.asarray(ideal_point, dtype=float)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle = np.column_stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)])
    return DescriptiveResult(
        name="preference_circles",
        value=circle,
        extra={"center": center.tolist(), "radius": float(radius)},
    )


pfcrc = preference_circles


def cheatsheet() -> str:
    return "preference_circles({}) -> Preference circles for visualization. 'Instant Transmission."
