# morie.fn — function file (hadesllm/morie)
"""Nambu-Goto string action."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def nambu_goto_action(
    X: np.ndarray | None = None,
    metric: np.ndarray | None = None,
    T: float = 1.0,
) -> DescriptiveResult:
    """Compute the Nambu-Goto action for a string worldsheet.

    .. math::

        S_{NG} = -T \\int d\\sigma\\,d\\tau\\,
                 \\sqrt{-\\det(h_{ab})}

    where :math:`h_{ab} = \\partial_a X^\\mu \\partial_b X_\\mu` is the
    induced metric on the worldsheet.

    :param X: Worldsheet embedding coordinates, shape (n_sigma, n_tau, d).
              If None, a sample flat string is used.
    :param metric: Pre-computed induced metric determinants, shape (n_sigma, n_tau).
    :param T: String tension.
    :return: DescriptiveResult with the action value.
    """
    if X is None and metric is None:
        n_s, n_t = 20, 20
        sigma = np.linspace(0, np.pi, n_s)
        tau = np.linspace(0, 2 * np.pi, n_t)
        S, Ta = np.meshgrid(sigma, tau, indexing="ij")
        X = np.stack([S, np.sin(S) * np.cos(Ta), np.sin(S) * np.sin(Ta)], axis=-1)

    if metric is not None:
        det_h = metric
    else:
        dX_ds = np.gradient(X, axis=0)
        dX_dt = np.gradient(X, axis=1)
        h00 = np.sum(dX_ds * dX_ds, axis=-1)
        h01 = np.sum(dX_ds * dX_dt, axis=-1)
        h11 = np.sum(dX_dt * dX_dt, axis=-1)
        det_h = h00 * h11 - h01**2

    integrand = np.sqrt(np.maximum(det_h, 0.0))
    action = -T * np.sum(integrand) * (np.pi / X.shape[0]) * (2 * np.pi / X.shape[1])
    return DescriptiveResult(
        name="nambu_goto_action",
        value=float(action),
        extra={"tension": T, "worldsheet_shape": X.shape, "det_h_mean": float(np.mean(det_h))},
    )


def cheatsheet() -> str:
    return "nambu_goto_action(X, metric, T) -> Nambu-Goto string action"


nmbu = nambu_goto_action
