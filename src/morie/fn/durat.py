# morie.fn — function file (hadesllm/morie)
"""Duration model (semiparametric hazard estimation)."""

from __future__ import annotations

import numpy as np


def durat(
    t: np.ndarray,
    event: np.ndarray | None = None,
    X: np.ndarray | None = None,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    n_grid: int = 100,
) -> dict:
    r"""
    Semiparametric duration/hazard model.

    Estimates the hazard function :math:`\lambda(t)` nonparametrically
    using kernel smoothing of the Nelson-Aalen estimator:

    .. math::

        \hat{\lambda}(t) = \frac{1}{h} \sum_{i: T_i \leq t}
        \frac{K((t - T_i)/h)}{Y(T_i)}

    where :math:`Y(t) = \sum_i \mathbf{1}[T_i \geq t]` is the at-risk
    count. If covariates X are provided, uses a multiplicative structure.

    Parameters
    ----------
    t : np.ndarray
        Duration/survival times (n,).
    event : np.ndarray or None
        Event indicator (n,): 1 = event, 0 = censored. Default all 1.
    X : np.ndarray or None
        Optional covariates (n, p) for multiplicative hazard.
    bandwidth : float or None
        Kernel bandwidth.
    kernel : str
        Kernel function.
    n_grid : int
        Grid size for hazard evaluation.

    Returns
    -------
    dict
        ``t_grid``, ``hazard``, ``survival``, ``cumulative_hazard``,
        ``bandwidth``, ``n_obs``, ``n_events``.

    References
    ----------
    Horowitz (2009). Ch 6 (transformation/duration models).
    """
    t = np.asarray(t, dtype=float).ravel()
    n = t.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations.")
    if np.any(t < 0):
        raise ValueError("Duration times must be non-negative.")

    if event is None:
        event = np.ones(n)
    else:
        event = np.asarray(event, dtype=float).ravel()
    if event.shape[0] != n:
        raise ValueError("event must have same length as t.")

    n_events = int(event.sum())

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    if bandwidth is None:
        bandwidth = _silverman_bw(t[event == 1]) if n_events > 1 else _silverman_bw(t)

    t_grid = np.linspace(t.min(), t.max(), n_grid)

    hazard = np.zeros(n_grid)
    for g in range(n_grid):
        at_risk = np.sum(t >= t_grid[g])
        if at_risk < 1:
            continue
        event_times = t[event == 1]
        u = (t_grid[g] - event_times) / bandwidth
        hazard[g] = k_fn(u).sum() / (at_risk * bandwidth)

    cumhaz = np.cumsum(hazard) * (t_grid[1] - t_grid[0]) if n_grid > 1 else hazard.copy()
    survival = np.exp(-cumhaz)

    return {
        "t_grid": t_grid.tolist(),
        "hazard": hazard.tolist(),
        "survival": survival.tolist(),
        "cumulative_hazard": cumhaz.tolist(),
        "bandwidth": float(bandwidth),
        "n_obs": n,
        "n_events": n_events,
    }


durat_fn = durat


def cheatsheet() -> str:
    return "durat({t}) -> Semiparametric duration/hazard model."
