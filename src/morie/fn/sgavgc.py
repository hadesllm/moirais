"""Average covariance over a block."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def average_covariance_block(
    coords_block: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
) -> SpatialResult:
    r"""Compute average covariance within a block.

    .. math::

        \bar{C}(B, B) = \frac{1}{m^2} \sum_{i=1}^{m}\sum_{j=1}^{m} C(s_i - s_j)

    Parameters
    ----------
    coords_block : np.ndarray
        Points in the block, shape ``(m, 2)``.
    cov_model : str
        ``"exponential"`` or ``"gaussian"``.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the block-average covariance.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    bc = np.asarray(coords_block, dtype=np.float64)
    m = len(bc)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _cov(h):
        if cov_model == "gaussian":
            return sill * np.exp(-((h / rng) ** 2)) + nug * (h == 0)
        return sill * np.exp(-h / rng) + nug * (h == 0)

    dist = np.sqrt(((bc[:, None, :] - bc[None, :, :]) ** 2).sum(-1))
    C = _cov(dist)
    avg_cov = float(np.mean(C))

    return SpatialResult(
        name="average_covariance_block",
        statistic=avg_cov,
        p_value=None,
        extra={"n_points": m, "cov_matrix": C},
    )


sgavgc = average_covariance_block


def cheatsheet() -> str:
    return "average_covariance_block({}) -> Average covariance over a block."
