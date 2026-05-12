# morie.fn -- function file (hadesllm/morie)
"""Cutting lines plot data."""

from __future__ import annotations

from ._containers import DescriptiveResult


def plot_cutting_lines_data(normals, cutpoints, X) -> DescriptiveResult:
    """Prepare line segments and legislator points for cutting line plots.

    .. epigraph:: "I watched Jane die." -- Walter White, Breaking Bad
    """
    import numpy as np

    normals = np.asarray(normals, dtype=float)
    cutpoints = np.asarray(cutpoints, dtype=float)
    X = np.asarray(X, dtype=float)
    lines = []
    for j in range(len(cutpoints)):
        nv = normals[j] if normals.ndim == 2 else normals
        if abs(nv[1]) > 1e-12:
            s = -nv[0] / nv[1]
            b = cutpoints[j] / nv[1]
            x_vals = np.array([-1.0, 1.0])
            y_vals = s * x_vals + b
            lines.append({"slope": float(s), "intercept": float(b), "x": x_vals.tolist(), "y": y_vals.tolist()})
    return DescriptiveResult(
        name="plot_cutting_lines_data",
        value=float(len(lines)),
        extra={"lines": lines, "points": X.tolist(), "n_lines": len(lines)},
    )


plcut = plot_cutting_lines_data


def cheatsheet() -> str:
    return "plot_cutting_lines_data({}) -> Cutting lines plot data."
