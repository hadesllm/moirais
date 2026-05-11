"""Spatial flow/interaction model (gravity model)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spatial_flow(
    origins: np.ndarray,
    destinations: np.ndarray,
    distances: np.ndarray,
    flows: np.ndarray | None = None,
    beta: float = 1.0,
) -> SpatialResult:
    r"""
    Gravity model for spatial interaction/flow estimation.

    Predicts the flow :math:`T_{ij}` between origin *i* and destination *j*
    as a function of origin mass, destination mass, and distance:

    .. math::

        \hat{T}_{ij} = k \frac{O_i^{\alpha} D_j^{\gamma}}{d_{ij}^{\beta}}

    where :math:`k` is a proportionality constant, :math:`O_i` and :math:`D_j`
    are origin/destination masses, and :math:`\beta` is the distance-decay
    parameter. When observed ``flows`` are provided, the function estimates
    :math:`\alpha, \gamma, \beta` via log-linear OLS.

    Parameters
    ----------
    origins : np.ndarray
        (n,) mass/size of origin locations (e.g. population).
    destinations : np.ndarray
        (m,) mass/size of destination locations.
    distances : np.ndarray
        (n, m) distance matrix between origins and destinations.
    flows : np.ndarray or None
        (n, m) observed flow matrix. If None, predicts using uncalibrated
        gravity model with given ``beta``.
    beta : float
        Distance-decay exponent (default 1.0, used when flows is None).

    Returns
    -------
    SpatialResult
        statistic = total predicted flow, extra has ``predicted_flows``,
        ``params`` (if calibrated), ``r_squared`` (if calibrated).

    References
    ----------
    Wilson AG (1971). A family of spatial interaction models, and
    associated developments. *Environment and Planning A*, 3(1), 1--32.
    doi:10.1068/a030001

    Fotheringham AS, O'Kelly ME (1989). *Spatial Interaction Models:
    Formulations and Applications*. Kluwer, Dordrecht.

    Examples
    --------
    >>> import numpy as np
    >>> O = np.array([1000, 2000, 500], dtype=float)
    >>> D = np.array([1500, 800], dtype=float)
    >>> dist = np.array([[10, 20], [15, 5], [25, 10]], dtype=float)
    >>> res = spatial_flow(O, D, dist, beta=2.0)
    >>> res.statistic > 0
    True
    """
    origins = np.asarray(origins, dtype=np.float64).ravel()
    destinations = np.asarray(destinations, dtype=np.float64).ravel()
    distances = np.asarray(distances, dtype=np.float64)
    n = len(origins)
    m = len(destinations)
    if distances.shape != (n, m):
        raise ValueError(f"distances must be ({n}, {m}).")

    safe_dist = np.where(distances < 1e-10, 1e-10, distances)

    if flows is not None:
        flows = np.asarray(flows, dtype=np.float64)
        mask = (flows > 0) & (distances > 0)
        log_T = np.log(flows[mask])
        log_O = np.log(np.maximum(origins, 1e-10))
        log_D = np.log(np.maximum(destinations, 1e-10))
        log_d = np.log(safe_dist)

        O_mat = np.tile(log_O[:, None], (1, m))
        D_mat = np.tile(log_D[None, :], (n, 1))

        X = np.column_stack(
            [
                np.ones(mask.sum()),
                O_mat[mask],
                D_mat[mask],
                log_d[mask],
            ]
        )
        try:
            coef, residuals, _, _ = np.linalg.lstsq(X, log_T, rcond=None)
        except np.linalg.LinAlgError:
            coef = np.array([0.0, 1.0, 1.0, -1.0])
            residuals = np.array([])

        k_hat = np.exp(coef[0])
        alpha = coef[1]
        gamma = coef[2]
        beta_hat = -coef[3]

        pred = k_hat * (origins[:, None] ** alpha) * (destinations[None, :] ** gamma) / (safe_dist**beta_hat)

        ss_res = float(np.sum((flows[mask] - pred[mask]) ** 2))
        ss_tot = float(np.sum((flows[mask] - flows[mask].mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

        return SpatialResult(
            name="spatial_flow",
            statistic=float(pred.sum()),
            extra={
                "predicted_flows": pred,
                "params": {"k": k_hat, "alpha": alpha, "gamma": gamma, "beta": beta_hat},
                "r_squared": r2,
                "n_origins": n,
                "n_destinations": m,
            },
        )

    pred = (origins[:, None] * destinations[None, :]) / (safe_dist**beta)

    return SpatialResult(
        name="spatial_flow",
        statistic=float(pred.sum()),
        extra={"predicted_flows": pred, "beta": beta, "n_origins": n, "n_destinations": m},
    )


spflw = spatial_flow


def cheatsheet() -> str:
    return "spatial_flow({}) -> Spatial flow/interaction model (gravity model)."
