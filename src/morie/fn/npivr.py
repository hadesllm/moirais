# morie.fn -- function file (hadesllm/morie)
"""Nonparametric instrumental variables estimation."""

from __future__ import annotations

import numpy as np


def npivr(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x: np.ndarray | None = None,
    *,
    bandwidth_dz: float | None = None,
    bandwidth_z: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Nonparametric instrumental variables (NPIV) estimation.

    Estimates the causal effect in the model:

    .. math::

        Y = g(D) + \varepsilon, \quad E[\varepsilon \mid Z] = 0

    using a two-stage nonparametric procedure:

    1. **Stage 1**: Estimate :math:`E[D \mid Z]` via Nadaraya-Watson
       to obtain :math:`\hat{D}`.
    2. **Stage 2**: Estimate :math:`E[Y \mid \hat{D}]` via
       Nadaraya-Watson to recover :math:`\hat{g}`.

    This is the nonparametric analogue of 2SLS. The local average
    derivative :math:`\hat{g}'` provides the average structural
    derivative (ASD).

    Parameters
    ----------
    y : np.ndarray
        Outcome vector (n,).
    d : np.ndarray
        Endogenous treatment (n,).
    z : np.ndarray
        Instrument (n,).
    x : np.ndarray or None
        Exogenous covariates (n,) or None. If provided, partialled out
        from y, d, and z via NW regression before the two-stage procedure.
    bandwidth_dz : float or None
        Bandwidth for first-stage (D on Z). If None, Silverman's rule.
    bandwidth_z : float or None
        Bandwidth for second-stage (Y on D_hat). If None, Silverman's rule.
    kernel : str
        Kernel: ``'gaussian'``, ``'epanechnikov'``, ``'uniform'``.

    Returns
    -------
    dict
        Keys: ``d_hat`` (first-stage fitted values), ``g_hat`` (structural
        function values), ``avg_deriv`` (average structural derivative),
        ``se_avg_deriv``, ``bandwidth_dz``, ``bandwidth_z``, ``n_obs``.

    References
    ----------
    Newey, W. K. & Powell, J. L. (2003). Instrumental variable estimation
        of nonparametric models. Econometrica, 71(5), 1565-1578.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 6.
    """
    y = np.asarray(y, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    n = y.shape[0]
    if d.shape[0] != n or z.shape[0] != n:
        raise ValueError("y, d, z must have same length.")
    if n < 10:
        raise ValueError("Need at least 10 observations.")

    from morie.fn.nwker import _silverman_bw, nwker

    if x is not None:
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != n:
            raise ValueError("x must have same length as y.")
        y_on_x = nwker(x, y, kernel=kernel)["y_hat"]
        d_on_x = nwker(x, d, kernel=kernel)["y_hat"]
        z_on_x = nwker(x, z, kernel=kernel)["y_hat"]
        y = y - y_on_x
        d = d - d_on_x
        z = z - z_on_x

    bw1 = bandwidth_dz if bandwidth_dz is not None else _silverman_bw(z)
    stage1 = nwker(z, d, bandwidth=bw1, kernel=kernel)
    d_hat = stage1["y_hat"]

    bw2 = bandwidth_z if bandwidth_z is not None else _silverman_bw(d_hat)
    stage2 = nwker(d_hat, y, bandwidth=bw2, kernel=kernel)
    g_hat = stage2["y_hat"]

    eps = bw2 * 0.01
    d_plus = d_hat + eps
    g_plus = nwker(d_hat, y, x_eval=d_plus, bandwidth=bw2, kernel=kernel)["y_hat"]
    g_prime = (g_plus - g_hat) / eps
    avg_deriv = float(np.nanmean(g_prime))

    resid = y - g_hat
    psi = g_prime * resid
    se_avg_deriv = float(np.nanstd(psi, ddof=1) / np.sqrt(n))

    return {
        "d_hat": d_hat,
        "g_hat": g_hat,
        "avg_deriv": avg_deriv,
        "se_avg_deriv": se_avg_deriv,
        "bandwidth_dz": bw1,
        "bandwidth_z": bw2,
        "n_obs": n,
    }


npivr_fn = npivr


def cheatsheet() -> str:
    return "npivr({y, d, z}) -> Nonparametric instrumental variables estimation."
