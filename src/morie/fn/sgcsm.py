"""Conditional simulation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def conditional_simulation(
    Z: np.ndarray,
    coords: np.ndarray,
    target_coords: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    n_sims: int = 10,
    seed: int = 42,
) -> SpatialResult:
    r"""Conditional simulation via kriging + unconditional sims.

    .. math::

        Z^{cs}(s) = Z^{u}(s) + [Z^{ok}(s) - Z^{ok,u}(s)]

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    target_coords : np.ndarray
        Target coordinates, shape ``(m, 2)``.
    cov_model : str
        Covariance model.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    n_sims : int
        Number of conditional realizations.
    seed : int
        RNG seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean of first conditional sim.
        ``extra`` has ``simulations`` shape ``(n_sims, m)``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.

    .. epigraph::

    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    tc = np.asarray(target_coords, dtype=np.float64)
    n = len(Z)
    m = len(tc)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    r = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    def _cov(h):
        if cov_model == "gaussian":
            return sill * np.exp(-((h / r) ** 2)) + nug * (h == 0)
        return sill * np.exp(-h / r) + nug * (h == 0)

    all_coords = np.vstack([coords, tc])
    N = n + m
    dist_all = np.sqrt(((all_coords[:, None, :] - all_coords[None, :, :]) ** 2).sum(-1))
    C_all = _cov(dist_all) + 1e-10 * np.eye(N)
    L = np.linalg.cholesky(C_all)

    dist_obs = dist_all[:n, :n]
    C_obs = _cov(dist_obs)

    def _ok(z_obs, c_obs, target_c):
        nn = len(z_obs)
        G = sill + nug - c_obs
        A = np.zeros((nn + 1, nn + 1))
        A[:nn, :nn] = G
        A[:nn, nn] = 1.0
        A[nn, :nn] = 1.0
        preds = np.empty(len(target_c))
        for j, t in enumerate(target_c):
            d0 = np.sqrt(((coords - t) ** 2).sum(-1))
            g0 = sill + nug - _cov(d0)
            b = np.zeros(nn + 1)
            b[:nn] = g0
            b[nn] = 1.0
            lam = np.linalg.solve(A, b)
            preds[j] = lam[:nn] @ z_obs
        return preds

    z_ok = _ok(Z, C_obs, tc)

    sims = np.empty((n_sims, m))
    for k in range(n_sims):
        w = rng.standard_normal(N)
        uncond = L @ w
        z_u_obs = uncond[:n]
        z_u_target = uncond[n:]
        z_u_ok = _ok(z_u_obs, C_obs, tc)
        sims[k] = z_u_target + (z_ok - z_u_ok)

    return SpatialResult(
        name="conditional_simulation",
        statistic=float(np.mean(sims[0])),
        p_value=None,
        extra={"simulations": sims},
    )


sgcsm = conditional_simulation


def cheatsheet() -> str:
    return "conditional_simulation({}) -> Conditional simulation."
