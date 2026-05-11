"""Single-index model via average derivative estimation."""

from __future__ import annotations

import numpy as np


def siavg(
    y: np.ndarray,
    X: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Single-index model estimation via average derivative (ADE).

    Uses the Powell-Stock-Stoker (1989) average derivative estimator
    to identify the index coefficient :math:`\beta` (up to scale) in
    :math:`E[Y|X] = G(X'\beta)`:

    .. math::

        \hat{\delta} = E[\nabla m(X)]
        \propto \beta

    Then normalises :math:`\|\hat{\beta}\| = 1`.

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Covariates (n, p).
    bandwidth : float or None
        Kernel bandwidth. If None, Silverman's rule.
    kernel : str
        Kernel function.

    Returns
    -------
    dict
        ``beta`` (normalised index coefficients), ``index``
        (:math:`X\hat{\beta}`), ``avg_derivative``, ``se``,
        ``n_obs``, ``p``.

    References
    ----------
    Powell, Stock & Stoker (1989). Econometrica, 57, 1403-1430.
    Horowitz (2009). Ch 2.
    """
    from morie.fn.avgde import avgde

    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != X rows {n}.")
    if p < 2:
        raise ValueError("Need p >= 2 covariates for single-index identification.")

    res = avgde(y, X, bandwidth=bandwidth, kernel=kernel)
    delta = np.asarray(res["avg_derivative"])

    norm = np.linalg.norm(delta)
    if norm < 1e-15:
        beta = np.zeros(p)
    else:
        beta = delta / norm

    index = X @ beta

    return {
        "beta": beta.tolist(),
        "index": index.tolist(),
        "avg_derivative": res["avg_derivative"],
        "se": res["se"],
        "n_obs": n,
        "p": p,
    }


siavg_fn = siavg


def cheatsheet() -> str:
    return "siavg({y, X}) -> Single-index model via average derivative."
