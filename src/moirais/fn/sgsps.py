"""Spectral (FFT-based) Gaussian random field simulation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spectral_grf_sim(
    coords: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    n_sims: int = 1,
    seed: int = 42,
) -> SpatialResult:
    r"""Simulate a GRF using the spectral (Fourier) method.

    Generates the field on a regular grid via circulant embedding
    and FFT.

    Parameters
    ----------
    coords : np.ndarray
        Grid coordinates, shape ``(n, 2)``. Must be on a regular grid.
    cov_model : str
        ``"exponential"`` or ``"gaussian"``.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    n_sims : int
        Number of realizations.
    seed : int
        RNG seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean of first realization.
        ``extra`` has ``simulations``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.

    .. epigraph::

        "You don't need a reason to help people." -- Zidane, FF9
    """
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = params.get("sill", 1.0)
    r = params.get("range", 1.0)
    nug = params.get("nugget", 0.0)

    ux = np.unique(np.round(coords[:, 0], 8))
    uy = np.unique(np.round(coords[:, 1], 8))
    nx, ny = len(ux), len(uy)

    dx = np.arange(nx) * (ux[1] - ux[0]) if nx > 1 else np.array([0.0])
    dy = np.arange(ny) * (uy[1] - uy[0]) if ny > 1 else np.array([0.0])
    DX, DY = np.meshgrid(dx, dy, indexing="ij")
    H = np.sqrt(DX**2 + DY**2)

    if cov_model == "gaussian":
        C_grid = sill * np.exp(-((H / r) ** 2))
    else:
        C_grid = sill * np.exp(-H / r)

    Nx, Ny = 2 * nx, 2 * ny
    C_embed = np.zeros((Nx, Ny))
    C_embed[:nx, :ny] = C_grid

    S = np.real(np.fft.fft2(C_embed))
    S = np.maximum(S, 0.0)

    sims = np.empty((n_sims, n))
    for k in range(n_sims):
        noise = rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))
        field = np.real(np.fft.ifft2(np.sqrt(S) * noise))[:nx, :ny]
        sims[k] = field.ravel()[:n]

    return SpatialResult(
        name="spectral_grf_sim",
        statistic=float(np.mean(sims[0])),
        p_value=None,
        extra={"simulations": sims},
    )


sgsps = spectral_grf_sim


def cheatsheet() -> str:
    return "spectral_grf_sim({}) -> Spectral (FFT-based) Gaussian random field simulation."
