"""Differential equation spatio-temporal covariance"""

import numpy as np

from ._containers import SpatialResult


def st_diff_equation(coords=None, times=None, data=None, *, n=30, diffusion=0.1):
    """Spatio-temporal covariance via diffusion differential equation.

    Models spatio-temporal dependence through a diffusion PDE:
    dZ/dt = D * nabla^2(Z) + noise

    Returns
    -------
    SpatialResult
    """
    if coords is None:
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 100, (n, 2))
    if times is None:
        times = np.arange(n, dtype=float)
    if data is None:
        data = np.random.default_rng(1).standard_normal(n)
    coords = np.asarray(coords, dtype=float)
    times = np.asarray(times, dtype=float)
    data = np.asarray(data, dtype=float)
    n_pts = len(data)
    D_space = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    D_time = np.abs(times[:, None] - times[None, :])
    C_st = np.exp(-D_space / (4 * diffusion * (D_time + 1e-10)))
    np.fill_diagonal(C_st, 1.0)
    stat = float(np.mean(C_st[np.triu_indices(n_pts, k=1)]))
    return SpatialResult(
        name="Diffusion equation ST covariance",
        statistic=stat,
        extra={"diffusion": diffusion, "n": n_pts, "mean_cov": stat},
    )


short = "stdea"
alias = "st_diff_equation"
quote = "A journey of a thousand miles begins with a single step. — Lao Tzu"
st_diff_equation = st_diff_equation


def cheatsheet() -> str:
    return "st_diff_equation({}) -> Differential equation spatio-temporal covariance"
