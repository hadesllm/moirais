# morie.fn -- function file (hadesllm/morie)
"""Modified gamma kernel density for positive-support data."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kgamm"]


def kgamm(
    data: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bw: float | None = None,
    n_grid: int = 512,
) -> dict:
    r"""
    Modified gamma kernel density estimator for positive data.

    Uses asymmetric gamma kernels that naturally respect the boundary at zero:

    .. math::

        \hat{f}(x) = \frac{1}{n} \sum_{i=1}^{n}
        \frac{x^{X_i / b}\, e^{-x/b}}{b^{X_i/b + 1}\,\Gamma(X_i/b + 1)}

    where *b* is the bandwidth parameter.

    Parameters
    ----------
    data : np.ndarray
        1-d array of strictly positive observations.
    x_eval : np.ndarray or None
        Positive grid points for evaluation.
    bw : float or None
        Bandwidth parameter *b*. If None, a rule of thumb is used.
    n_grid : int
        Grid size when *x_eval* is None.

    Returns
    -------
    dict
        ``x_eval``, ``density``, ``bw``.

    Raises
    ------
    ValueError
        If data contains non-positive values.

    References
    ----------
    Chen, S. X. (2000). Probability density function estimation using gamma
        kernels. *Annals of the Institute of Statistical Mathematics*,
        52(3), 471-480.
    """
    from scipy.special import gammaln

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    if np.any(data <= 0):
        raise ValueError("All observations must be strictly positive.")

    if bw is None:
        sigma = np.std(data, ddof=1)
        bw = max(sigma * n ** (-0.4), 1e-10)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    if x_eval is None:
        x_eval = np.linspace(1e-6, data.max() + 3 * np.std(data, ddof=1), n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    x_eval = np.maximum(x_eval, 1e-300)

    alpha = data[None, :] / bw + 1.0
    log_density = np.zeros(len(x_eval))
    for j, xj in enumerate(x_eval):
        log_k = (alpha[0, :] - 1) * np.log(xj) - xj / bw - alpha[0, :] * np.log(bw) - gammaln(alpha[0, :])
        log_density[j] = np.log(np.mean(np.exp(log_k - log_k.max())) + 1e-300) + log_k.max()

    density = np.exp(log_density)
    density = np.maximum(density, 0.0)

    return RichResult(payload={"x_eval": x_eval, "density": density, "bw": bw})


def cheatsheet() -> str:
    return "kgamm({data}) -> Gamma kernel density for positive data."
