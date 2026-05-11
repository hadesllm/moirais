# morie.fn — function file (hadesllm/morie)
"""Coombs mesh visualization data."""

from __future__ import annotations

from ._containers import DescriptiveResult


def plot_coombs_data(mesh_result) -> DescriptiveResult:
    """Extract grid data from a Coombs mesh result for visualization.

    .. epigraph:: "Ozymandias." -- Breaking Bad
    """
    import numpy as np

    if hasattr(mesh_result, "extra"):
        extra = mesh_result.extra
    else:
        extra = mesh_result
    grid = np.asarray(extra.get("yea_fraction_grid", []))
    gx = np.asarray(extra.get("grid_x", []))
    gy = np.asarray(extra.get("grid_y", []))
    return DescriptiveResult(
        name="plot_coombs_data",
        value=float(np.mean(grid)) if grid.size > 0 else 0.0,
        extra={
            "grid": grid,
            "grid_x": gx,
            "grid_y": gy,
            "shape": list(grid.shape) if grid.ndim >= 2 else [],
        },
    )


plcmb = plot_coombs_data


def cheatsheet() -> str:
    return "plot_coombs_data({}) -> Coombs mesh visualization data."
