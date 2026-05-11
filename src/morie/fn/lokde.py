# morie.fn — function file (hadesllm/morie)
"""Local polynomial density estimator."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["lokde"]


def lokde(
    data: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bw: float | None = None,
    degree: int = 2,
    n_grid: int = 512,
) -> dict:
    r"""
    Local polynomial density estimator (LPDE).

    Fits a local polynomial of degree *p* to the log-density at each
    evaluation point, which automatically corrects boundary bias without
    explicit boundary treatment.

    The estimator solves the local likelihood:

    .. math::

        \max_{\beta} \sum_{i=1}^{n} K_h(x - X_i) \sum_{j=0}^{p}
        \beta_j (X_i - x)^j
        - n \int K_h(x - t)\exp\!\left(\sum_{j=0}^{p}\beta_j(t-x)^j\right)dt

    and the density estimate is :math:`\hat{f}(x) = \exp(\hat\beta_0)`.

    A simpler practical approach is used here: local polynomial
    regression on an initial pilot density estimate.

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations.
    x_eval : np.ndarray or None
        Grid of evaluation points.
    bw : float or None
        Bandwidth. Silverman's rule if None.
    degree : int
        Polynomial degree (default 2).
    n_grid : int
        Grid size when *x_eval* is None.

    Returns
    -------
    dict
        ``x_eval``, ``density``, ``bw``, ``degree``.

    References
    ----------
    Loader, C. R. (1996). Local likelihood density estimation. *Annals of
        Statistics*, 24(4), 1602-1618.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    sigma = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    if bw is None:
        bw = max(0.9 * s * n ** (-0.2), 1e-10)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    if x_eval is None:
        lo = data.min() - 3 * bw
        hi = data.max() + 3 * bw
        x_eval = np.linspace(lo, hi, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    u_pilot = (x_eval[:, None] - data[None, :]) / bw
    pilot = np.exp(-0.5 * u_pilot ** 2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    pilot = np.maximum(pilot, 1e-300)
    log_pilot = np.log(pilot)

    density = np.empty_like(x_eval)
    for i, x0 in enumerate(x_eval):
        w = np.exp(-0.5 * ((x_eval - x0) / bw) ** 2)
        w /= w.sum() + 1e-300
        dx = x_eval - x0
        X = np.column_stack([dx ** j for j in range(degree + 1)])
        Xw = X * w[:, None]
        try:
            beta = np.linalg.lstsq(Xw, log_pilot * w, rcond=None)[0]
            density[i] = np.exp(beta[0])
        except np.linalg.LinAlgError:
            density[i] = pilot[i]

    area = np.trapezoid(density, x_eval)
    if area > 0:
        density /= area

    return RichResult(payload={"x_eval": x_eval, "density": density, "bw": bw, "degree": degree})


def cheatsheet() -> str:
    return "lokde({data}) -> Local polynomial density estimator."
