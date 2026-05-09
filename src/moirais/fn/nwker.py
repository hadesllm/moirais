# moirais.fn — function file (hadesllm/moirais)
"""Nadaraya-Watson kernel regression estimator."""

from __future__ import annotations

import numpy as np


def nwker(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Nadaraya-Watson kernel regression estimator.

    Estimates the conditional mean :math:`m(x) = E[Y \mid X = x]` using:

    .. math::

        \hat{m}(x) = \frac{\sum_{i=1}^{n} K_h(x - X_i)\, Y_i}
                           {\sum_{i=1}^{n} K_h(x - X_i)}

    where :math:`K_h(\cdot) = K(\cdot / h) / h` is a scaled kernel with
    bandwidth :math:`h`.

    Parameters
    ----------
    x : np.ndarray
        1-d array of predictor values (n,).
    y : np.ndarray
        1-d array of response values (n,).
    x_eval : np.ndarray or None
        Points at which to evaluate the estimator. Defaults to ``x``.
    bandwidth : float or None
        Kernel bandwidth. If None, uses Silverman's rule of thumb.
    kernel : str
        Kernel function: ``'gaussian'``, ``'epanechnikov'``, or ``'uniform'``.

    Returns
    -------
    dict
        Keys: ``x_eval`` (evaluation points), ``y_hat`` (fitted values),
        ``bandwidth`` (bandwidth used), ``kernel`` (kernel name), ``n_obs``.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or unknown kernel.

    References
    ----------
    Nadaraya, E. A. (1964). On estimating regression. Theory of Probability
        and its Applications, 9(1), 141-142.
    Watson, G. S. (1964). Smooth regression analysis. Sankhya A, 26, 359-372.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have same length, got {x.shape[0]} and {y.shape[0]}.")
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    valid_kernels = {"gaussian", "epanechnikov", "uniform"}
    if kernel not in valid_kernels:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from {valid_kernels}.")

    if x_eval is None:
        x_eval = x.copy()
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    if bandwidth is None:
        bandwidth = _silverman_bw(x)

    if bandwidth <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth}.")

    k_fn = _get_kernel(kernel)
    u = (x_eval[:, None] - x[None, :]) / bandwidth
    w = k_fn(u)
    denom = w.sum(axis=1)
    denom = np.where(denom == 0, 1.0, denom)
    y_hat = (w * y[None, :]).sum(axis=1) / denom

    return {
        "x_eval": x_eval,
        "y_hat": y_hat,
        "bandwidth": bandwidth,
        "kernel": kernel,
        "n_obs": n,
    }


def _silverman_bw(x: np.ndarray) -> float:
    """Silverman (1986) rule-of-thumb bandwidth."""
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 1.06 * min(s, iqr / 1.34) * n ** (-1 / 5)


def _get_kernel(name: str):
    """Return kernel function by name."""
    if name == "gaussian":
        return lambda u: np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    elif name == "epanechnikov":
        return lambda u: np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
    elif name == "uniform":
        return lambda u: np.where(np.abs(u) <= 1, 0.5, 0.0)


nwker_fn = nwker


def cheatsheet() -> str:
    return "nwker({x, y}) -> Nadaraya-Watson kernel regression estimator."
