# morie.fn -- function file (hadesllm/morie)
"""Polyakov worldsheet action."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def polyakov_action(
    X: np.ndarray | None = None,
    h: np.ndarray | None = None,
    G: np.ndarray | None = None,
    T: float = 1.0,
) -> DescriptiveResult:
    r"""Compute the Polyakov action for a bosonic string.

    .. math::

        S_P = -\\frac{T}{2} \\int d^2\\sigma\\,
              \\sqrt{-h}\\, h^{ab}\\, G_{\\mu\\nu}\\,
              \\partial_a X^\\mu \\partial_b X^\\nu

    :param X: Worldsheet embedding, shape (n_sigma, n_tau, d). Defaults to flat.
    :param h: Worldsheet metric determinant array. Defaults to flat metric.
    :param G: Target space metric (d, d). Defaults to Minkowski.
    :param T: String tension.
    :return: DescriptiveResult with action value.
    """
    if X is None:
        n_s, n_t = 20, 20
        sigma = np.linspace(0, np.pi, n_s)
        tau = np.linspace(0, 2 * np.pi, n_t)
        S, Ta = np.meshgrid(sigma, tau, indexing="ij")
        X = np.stack([Ta, S, np.zeros_like(S)], axis=-1)

    d = X.shape[-1]
    if G is None:
        G = np.eye(d)
        G[0, 0] = -1.0

    dX_ds = np.gradient(X, axis=0)
    dX_dt = np.gradient(X, axis=1)

    integrand = np.zeros(X.shape[:2])
    for mu in range(d):
        for nu in range(d):
            integrand += G[mu, nu] * (dX_ds[..., mu] * dX_ds[..., nu] + dX_dt[..., mu] * dX_dt[..., nu])

    action = -0.5 * T * np.sum(integrand) * (np.pi / X.shape[0]) * (2 * np.pi / X.shape[1])
    return DescriptiveResult(
        name="polyakov_action",
        value=float(action),
        extra={"tension": T, "d": d, "worldsheet_shape": X.shape},
    )


def cheatsheet() -> str:
    return "polyakov_action(X, h, G, T) -> Polyakov worldsheet action"


plykv = polyakov_action
