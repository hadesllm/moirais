# moirais.fn — function file (hadesllm/moirais)
"""Hadamard differentiability check via numerical perturbation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def hadamard_differentiability(
    phi: Callable[[np.ndarray], float],
    theta: np.ndarray,
    *,
    h: np.ndarray | None = None,
    t_values: np.ndarray | None = None,
    tol: float = 1e-3,
) -> dict:
    r"""
    Numerical check for Hadamard differentiability of a functional.

    A map :math:`\phi: \mathbb{D} \to \mathbb{E}` is Hadamard
    differentiable at :math:`\theta` tangentially to :math:`\mathbb{D}_0`
    if there exists a continuous linear map :math:`\phi'_\theta` such that:

    .. math::

        \frac{\phi(\theta + t_n h_n) - \phi(\theta)}{t_n}
        \to \phi'_\theta(h)

    for all sequences :math:`t_n \downarrow 0` and :math:`h_n \to h`.

    This function checks the condition numerically by evaluating the
    difference quotient at decreasing :math:`t` values and verifying
    convergence.

    :param phi: Functional mapping arrays to scalars.
    :param theta: Point of evaluation (1-D array).
    :param h: Direction of perturbation. Default: unit random direction.
    :param t_values: Decreasing sequence of perturbation sizes. Default
        geometric: ``[0.1, 0.01, 0.001, 0.0001, 1e-5]``.
    :param tol: Convergence tolerance for successive quotients. Default 1e-3.
    :return: dict with ``is_differentiable`` (bool), ``quotients`` (list
        of difference quotients), ``derivative_estimate`` (last quotient),
        ``convergence_ratios``.
    :raises ValueError: If theta is empty.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Ch. 9 (Hadamard differentiability).
        Springer.
    van der Vaart, A.W. & Wellner, J.A. (1996). *Weak Convergence and
        Empirical Processes*, Ch. 3.9. Springer.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size == 0:
        raise ValueError("theta must be non-empty.")

    if h is None:
        rng = np.random.default_rng(42)
        h = rng.standard_normal(theta.shape)
        h = h / np.linalg.norm(h)
    else:
        h = np.asarray(h, dtype=float).ravel()

    if t_values is None:
        t_values = np.array([0.1, 0.01, 0.001, 0.0001, 1e-5])
    else:
        t_values = np.asarray(t_values, dtype=float)

    phi_theta = phi(theta)

    quotients = []
    for t in t_values:
        perturbed = phi(theta + t * h)
        quotients.append((perturbed - phi_theta) / t)

    ratios = []
    for i in range(1, len(quotients)):
        diff = abs(quotients[i] - quotients[i - 1])
        ratios.append(float(diff))

    is_diff = len(ratios) > 0 and all(r < tol for r in ratios[-2:]) if len(ratios) >= 2 else (
        len(ratios) > 0 and ratios[-1] < tol
    )

    return {
        "is_differentiable": bool(is_diff),
        "quotients": [float(q) for q in quotients],
        "derivative_estimate": float(quotients[-1]) if quotients else float("nan"),
        "convergence_ratios": ratios,
        "t_values": t_values.tolist(),
        "method": "Hadamard differentiability (numerical)",
    }


hddif = hadamard_differentiability


def cheatsheet() -> str:
    return "hadamard_differentiability(phi, theta) -> Hadamard differentiability check."
