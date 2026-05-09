"""Ordinary kriging interpolation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def ordinary_kriging(
    Z: np.ndarray,
    coords: np.ndarray,
    target: np.ndarray,
    vario_model: str = "exponential",
    vario_params: dict | None = None,
) -> SpatialResult:
    r"""Ordinary kriging with unknown constant mean.

    Solves the kriging system with a Lagrange multiplier for the
    unbiasedness constraint :math:`\sum \lambda_i = 1`.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target : np.ndarray
        Prediction location, shape ``(2,)`` or ``(m, 2)``.
    vario_model : str
        ``"exponential"``, ``"gaussian"``, or ``"spherical"``.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the predicted value; ``extra`` has
        ``predictions``, ``variances``, ``weights``, ``lagrange``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "A wizard is never late." -- Gandalf, LOTR
    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    n = len(Z)
    params = vario_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        if vario_model == "spherical":
            hr = np.minimum(h / rng, 1.0)
            return nug + sill * (1.5 * hr - 0.5 * hr**3) * (h > 0)
        if vario_model == "gaussian":
            return nug + sill * (1.0 - np.exp(-((h / rng) ** 2))) * (h > 0)
        return nug + sill * (1.0 - np.exp(-h / rng)) * (h > 0)

    dist_obs = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    G = _gamma(dist_obs)

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0

    preds, varis, all_w, all_mu = [], [], [], []
    for t in target:
        d0 = np.sqrt(((coords - t) ** 2).sum(-1))
        g0 = _gamma(d0)
        b = np.zeros(n + 1)
        b[:n] = g0
        b[n] = 1.0
        lam = np.linalg.solve(A, b)
        w = lam[:n]
        mu = lam[n]
        pred = w @ Z
        var = w @ g0 + mu
        preds.append(float(pred))
        varis.append(float(max(var, 0.0)))
        all_w.append(w)
        all_mu.append(float(mu))

    return SpatialResult(
        name="ordinary_kriging",
        statistic=preds[0],
        p_value=None,
        extra={
            "predictions": np.array(preds),
            "variances": np.array(varis),
            "weights": all_w,
            "lagrange": all_mu,
        },
    )


sgok = ordinary_kriging


def cheatsheet() -> str:
    return "ordinary_kriging({}) -> Ordinary kriging interpolation."
