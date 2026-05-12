# morie.fn -- function file (hadesllm/morie)
"""OC Coombs mesh predicted choice grid."""

from __future__ import annotations

from ._containers import DescriptiveResult


def oc_coombs_mesh(normals, cutpoints, grid_size=50) -> DescriptiveResult:
    """Build a 2D grid of predicted choices from cutting lines.

    .. epigraph:: "I won." -- Walter White, Breaking Bad
    """
    import numpy as np

    normals = np.asarray(normals, dtype=float)
    cutpoints = np.asarray(cutpoints, dtype=float)
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    n_bills = len(cutpoints)
    votes = np.zeros((grid.shape[0], n_bills))
    for j in range(n_bills):
        n_vec = normals[j] if normals.ndim == 2 else normals
        proj = grid @ n_vec
        votes[:, j] = (proj >= cutpoints[j]).astype(float)
    yea_frac = np.mean(votes, axis=1).reshape(grid_size, grid_size)
    return DescriptiveResult(
        name="oc_coombs_mesh",
        value=float(np.mean(yea_frac)),
        extra={
            "yea_fraction_grid": yea_frac,
            "grid_x": x,
            "grid_y": y,
            "grid_size": grid_size,
            "n_bills": n_bills,
        },
    )


ocmsh = oc_coombs_mesh


def cheatsheet() -> str:
    return "oc_coombs_mesh({}) -> OC Coombs mesh predicted choice grid."
