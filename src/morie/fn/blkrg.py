# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Block kriging — areal prediction (Schabenberger & Gotway Ch 5)."""

import numpy as np
from scipy.spatial.distance import cdist


def blkrg(
    coords: np.ndarray,
    values: np.ndarray,
    block_coords: np.ndarray,
    *,
    sill: float = 1.0,
    range_param: float = 1.0,
    nugget: float = 0.0,
    n_discretise: int = 25,
    seed: int | None = None,
) -> dict:
    """
    Block kriging for areal mean prediction.

    Averages the kriging predictions over a block (area) defined by
    the centroid and a regular discretisation grid.

    :param coords: Point observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param block_coords: Block centroids (m, 2).
    :param sill: Variogram sill.
    :param range_param: Variogram range.
    :param nugget: Nugget variance.
    :param n_discretise: Points to discretise each block.
    :param seed: Random seed for discretisation.
    :return: dict with ``predictions``, ``variances``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    block_coords = np.asarray(block_coords, dtype=float)
    n = len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")
    if block_coords.ndim == 1:
        block_coords = block_coords.reshape(1, -1)

    rng = np.random.default_rng(seed)
    D = cdist(coords, coords)
    C = sill * np.exp(-D / (range_param + 1e-12))
    np.fill_diagonal(C, C.diagonal() + nugget)

    C_aug = np.zeros((n + 1, n + 1))
    C_aug[:n, :n] = C
    C_aug[:n, n] = 1.0
    C_aug[n, :n] = 1.0

    half_size = range_param * 0.5
    m = len(block_coords)
    preds = np.zeros(m)
    variances = np.zeros(m)

    for j in range(m):
        offsets = rng.uniform(-half_size, half_size, (n_discretise, 2))
        disc_pts = block_coords[j] + offsets

        c0_all = np.zeros(n)
        for pt in disc_pts:
            d0 = cdist(coords, pt.reshape(1, -1)).ravel()
            c0_all += sill * np.exp(-d0 / (range_param + 1e-12))
        c0_avg = c0_all / n_discretise

        rhs = np.append(c0_avg, 1.0)
        try:
            lam = np.linalg.solve(C_aug, rhs)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(C_aug, rhs, rcond=None)[0]
        preds[j] = float(lam[:n] @ values)
        variances[j] = max(float(sill + nugget - lam @ rhs), 0.0)

    return {
        "predictions": preds,
        "variances": variances,
        "n": n,
        "n_blocks": m,
        "n_discretise": n_discretise,
    }


blkrg_fn = blkrg


def cheatsheet() -> str:
    return "blkrg({}) -> Block kriging for areal prediction."
