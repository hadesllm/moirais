# morie.fn -- function file (hadesllm/morie)
"""Clark-Evans nearest neighbor index."""

import numpy as np
from scipy import stats as sp_stats

from ._containers import SpatialResult


def nearest_neighbor_index(points: np.ndarray) -> SpatialResult:
    r"""
    Compute the Clark-Evans nearest neighbor index.

    .. math::

        R = \\frac{\\bar{d}_{obs}}{\\bar{d}_{exp}}

    where :math:`\\bar{d}_{exp} = 0.5 / \\sqrt{\\lambda}`.
    R < 1 indicates clustering, R > 1 indicates regularity.

    :param points: (n, 2) array of coordinates.
    :return: SpatialResult with R statistic and z-test p-value.

    References
    ----------
    Clark PJ, Evans FC (1954). Distance to nearest neighbor as a
    measure of spatial relationships in populations.
    Ecology, 35(4), 445-453.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (n, 2).")
    n = pts.shape[0]
    dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dmat, np.inf)
    nn_dists = dmat.min(axis=1)
    d_obs = float(nn_dists.mean())
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    area = x_range * y_range if x_range > 0 and y_range > 0 else 1.0
    lam = n / area
    d_exp = 0.5 / np.sqrt(lam)
    R = d_obs / d_exp if d_exp > 0 else 1.0
    se = 0.26136 / np.sqrt(n * lam)
    z = (d_obs - d_exp) / se if se > 0 else 0.0
    pval = float(2 * sp_stats.norm.sf(abs(z)))
    return SpatialResult(
        name="nearest_neighbor_index",
        statistic=float(R),
        p_value=pval,
        expected=float(d_exp),
        extra={"d_obs": d_obs, "d_exp": float(d_exp), "z": float(z), "n": n, "area": area},
    )


nnidx = nearest_neighbor_index


def cheatsheet() -> str:
    return "nearest_neighbor_index({}) -> Clark-Evans nearest neighbor index."
