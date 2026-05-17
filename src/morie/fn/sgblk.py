"""Block kriging."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def block_kriging(
    Z: np.ndarray,
    coords: np.ndarray,
    block_coords: np.ndarray,
    vario_model: str = "exponential",
    vario_params: dict | None = None,
) -> SpatialResult:
    r"""Block kriging: predict average over a block.

    Predicts the block average :math:`\bar{Z}(B)` by averaging
    point kriging predictions over the block support.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    block_coords : np.ndarray
        Points discretizing the block, shape ``(m, 2)``.
    vario_model : str
        Variogram model.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the block average prediction.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    block_coords = np.asarray(block_coords, dtype=np.float64)
    n = len(Z)
    m = len(block_coords)
    params = vario_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        if vario_model == "gaussian":
            return nug + sill * (1.0 - np.exp(-((h / rng) ** 2))) * (h > 0)
        return nug + sill * (1.0 - np.exp(-h / rng)) * (h > 0)

    dist_obs = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    G = _gamma(dist_obs)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0

    g0_avg = np.zeros(n)
    for bc in block_coords:
        d0 = np.sqrt(((coords - bc) ** 2).sum(-1))
        g0_avg += _gamma(d0)
    g0_avg /= m

    b = np.zeros(n + 1)
    b[:n] = g0_avg
    b[n] = 1.0
    lam = np.linalg.solve(A, b)
    w = lam[:n]
    pred = w @ Z
    var = w @ g0_avg + lam[n]

    return SpatialResult(
        name="block_kriging",
        statistic=float(pred),
        p_value=None,
        extra={
            "variance": float(max(var, 0.0)),
            "weights": w,
            "n_block_points": m,
        },
    )


sgblk = block_kriging


def cheatsheet() -> str:
    return "block_kriging({}) -> Block kriging."
