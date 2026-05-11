# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Boundary-corrected kernel density estimation."""

from __future__ import annotations

import numpy as np


def bkde(
    x: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    lower: float | None = None,
    upper: float | None = None,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    method: str = "reflection",
    n_grid: int = 256,
) -> dict:
    r"""
    Boundary-corrected kernel density estimation.

    Standard KDE is biased near boundaries of the support. This uses
    one of:

    - ``'reflection'``: reflect data at boundaries (Silverman 1986).
    - ``'renormalization'``: renormalize kernel mass within support.

    Parameters
    ----------
    x : np.ndarray
        Data vector (n,).
    x_eval : np.ndarray or None
        Evaluation grid.
    lower, upper : float or None
        Boundary of the support. None means no boundary on that side.
    bandwidth : float or None
        If None, Silverman's rule.
    kernel : str
        ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.
    method : str
        ``'reflection'`` or ``'renormalization'``.
    n_grid : int
        Grid size when ``x_eval`` is None.

    Returns
    -------
    dict
        ``x_eval``, ``density``, ``bandwidth``, ``method``, ``n_obs``.

    References
    ----------
    Silverman (1986). Ch 2.10.
    Horowitz (2009). Ch 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 observations.")
    if method not in ("reflection", "renormalization"):
        raise ValueError("method must be 'reflection' or 'renormalization'.")

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    if bandwidth is None:
        bandwidth = _silverman_bw(x)

    if x_eval is None:
        lo = lower if lower is not None else x.min() - 3 * bandwidth
        hi = upper if upper is not None else x.max() + 3 * bandwidth
        x_eval = np.linspace(lo, hi, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    if method == "reflection":
        x_aug = x.copy()
        if lower is not None:
            x_aug = np.concatenate([x_aug, 2 * lower - x])
        if upper is not None:
            x_aug = np.concatenate([x_aug, 2 * upper - x])
        n_aug = len(x_aug)
        u = (x_eval[:, None] - x_aug[None, :]) / bandwidth
        density = k_fn(u).sum(axis=1) / (n * bandwidth)
    else:
        u = (x_eval[:, None] - x[None, :]) / bandwidth
        raw = k_fn(u).mean(axis=1) / bandwidth
        mass = np.ones(len(x_eval))
        if lower is not None or upper is not None:
            fine = 1000
            for idx, xp in enumerate(x_eval):
                lo_int = (lower - xp) / bandwidth if lower is not None else -10
                hi_int = (upper - xp) / bandwidth if upper is not None else 10
                grid_int = np.linspace(lo_int, hi_int, fine)
                _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
                mass[idx] = _trapz(k_fn(grid_int), grid_int)
        mass = np.maximum(mass, 1e-15)
        density = raw / mass

    return {
        "x_eval": x_eval.tolist(),
        "density": density.tolist(),
        "bandwidth": float(bandwidth),
        "method": method,
        "n_obs": n,
    }


bkde_fn = bkde


def cheatsheet() -> str:
    return "bkde({x}) -> Boundary-corrected KDE (reflection/renormalization)."
