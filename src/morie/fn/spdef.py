"""Spatial deformation model (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def spdef(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_dims: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int | None = None,
) -> dict:
    """
    Fit a spatial deformation model via multidimensional scaling.

    Maps geographic coordinates to a deformed space (D-space) where
    the covariance structure is approximately stationary. Uses classical
    MDS on a dissimilarity matrix derived from local variogram estimates.

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param n_dims: Dimensions of the deformed space.
    :param max_iter: Maximum MDS iterations (Smacof-like stress minimisation).
    :param tol: Convergence tolerance.
    :param seed: Random seed.
    :return: dict with ``deformed_coords``, ``stress``, ``dissimilarity``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Sampson, P. D. & Guttorp, P. (1992). Nonparametric estimation of
    nonstationary spatial covariance structure. *JASA*, 87(417), 108-119.

    Schabenberger & Gotway (2005), Ch. 8.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")

    geo_dists = squareform(pdist(coords))
    sq_diff = squareform(pdist(values.reshape(-1, 1)) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        diss = np.where(geo_dists > 0, np.sqrt(sq_diff / 2), 0.0)

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_dims))

    stress = np.inf
    for _ in range(max_iter):
        d_current = squareform(pdist(X))
        d_current_safe = np.where(d_current > 0, d_current, 1.0)
        B = -diss / d_current_safe
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))
        X_new = B @ X / n
        new_stress = np.sqrt(np.sum((diss - squareform(pdist(X_new))) ** 2) / np.sum(diss ** 2 + 1e-12))
        if abs(stress - new_stress) < tol:
            X = X_new
            stress = new_stress
            break
        X = X_new
        stress = new_stress

    return {
        "deformed_coords": X,
        "stress": float(stress),
        "dissimilarity": diss,
        "n_dims": n_dims,
        "n": n,
    }


spdef_fn = spdef


def cheatsheet() -> str:
    return "spdef({}) -> Spatial deformation model via MDS."
