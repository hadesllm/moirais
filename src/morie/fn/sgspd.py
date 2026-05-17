"""Space deformation non-stationary covariance"""

import numpy as np

from ._containers import SpatialResult


def space_deformation(coords=None, data=None, *, n=50, dims=2):
    """Space deformation model for non-stationary covariance.

    Transforms spatial coordinates via a smooth mapping f: R^d -> R^d
    such that the process is stationary in the deformed space.

    Returns
    -------
    SpatialResult
    """
    if coords is None:
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 100, (n, dims))
    if data is None:
        data = np.random.default_rng(1).standard_normal(len(coords))
    coords = np.asarray(coords, dtype=float)
    data = np.asarray(data, dtype=float)
    centroid = coords.mean(axis=0)
    diffs = coords - centroid
    dists = np.sqrt((diffs**2).sum(axis=1))
    scale = np.exp(-0.01 * dists)
    deformed = coords * scale[:, None]
    D_orig = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    D_def = np.sqrt(((deformed[:, None] - deformed[None, :]) ** 2).sum(axis=-1))
    mask = np.triu(np.ones_like(D_orig, dtype=bool), k=1)
    corr = float(np.corrcoef(D_orig[mask], D_def[mask])[0, 1])
    return SpatialResult(
        name="Space deformation",
        statistic=corr,
        extra={"n": len(coords), "dims": dims, "deformation_corr": corr},
    )


short = "sgspd"
alias = "space_deformation"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
space_deformation = space_deformation


def cheatsheet() -> str:
    return "space_deformation({}) -> Space deformation non-stationary covariance"
