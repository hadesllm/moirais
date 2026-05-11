"""Kriging-based conditional simulation correction."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def kriging_conditional_sim(
    Z: np.ndarray,
    coords: np.ndarray,
    unconditional_sims: np.ndarray,
    sim_coords: np.ndarray | None = None,
    vario_model: str = "exponential",
    vario_params: dict | None = None,
) -> SpatialResult:
    r"""Condition unconditional simulations via kriging correction.

    .. math::

        Z^{cs}_k(s) = Z^{u}_k(s) + [\hat{Z}(s) - \hat{Z}^{u}_k(s)]

    Parameters
    ----------
    Z : np.ndarray
        Conditioning data, shape ``(n,)``.
    coords : np.ndarray
        Data coordinates, shape ``(n, 2)``.
    unconditional_sims : np.ndarray
        Unconditional sims, shape ``(n_sims, m)`` or ``(n_sims, n)``.
    sim_coords : np.ndarray, optional
        Simulation coordinates, shape ``(m, 2)``. Defaults to ``coords``.
    vario_model : str
        Variogram model.
    vario_params : dict, optional
        ``{"sill", "range", "nugget"}``.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean of first conditioned sim.
        ``extra`` has ``conditioned_sims``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.

    .. epigraph::

        "I am the shadow, the true self." -- Shadow, Persona 4
    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    usims = np.asarray(unconditional_sims, dtype=np.float64)
    n = len(Z)

    if sim_coords is None:
        sim_coords = coords.copy()
    else:
        sim_coords = np.asarray(sim_coords, dtype=np.float64)
    m = len(sim_coords)

    params = vario_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _gamma(h):
        return nug + sill * (1.0 - np.exp(-h / rng)) * (h > 0)

    dist_obs = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    G = _gamma(dist_obs)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0

    def _krig(z_data, targets):
        preds = np.empty(len(targets))
        for j, t in enumerate(targets):
            d0 = np.sqrt(((coords - t) ** 2).sum(-1))
            g0 = _gamma(d0)
            b = np.zeros(n + 1)
            b[:n] = g0
            b[n] = 1.0
            lam = np.linalg.solve(A, b)
            preds[j] = lam[:n] @ z_data
        return preds

    z_ok = _krig(Z, sim_coords)

    n_sims = usims.shape[0]
    csims = np.empty_like(usims)
    for k in range(n_sims):
        u_at_obs = usims[k, :n] if usims.shape[1] >= n else usims[k]
        z_u_ok = _krig(u_at_obs[:n], sim_coords)
        csims[k] = usims[k] + (z_ok[: usims.shape[1]] - z_u_ok[: usims.shape[1]])

    return SpatialResult(
        name="kriging_conditional_sim",
        statistic=float(np.mean(csims[0])),
        p_value=None,
        extra={"conditioned_sims": csims},
    )


sgkcs = kriging_conditional_sim


def cheatsheet() -> str:
    return "kriging_conditional_sim({}) -> Kriging-based conditional simulation correction."
