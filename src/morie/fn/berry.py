# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Berry geometric phase."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def berry_phase(
    H_params: np.ndarray | None = None,
    path: np.ndarray | None = None,
    n_points: int = 100,
) -> DescriptiveResult:
    r"""Compute the Berry phase for a parametric Hamiltonian.

    For a 2-level system with H(theta, phi) = B(sin(theta)cos(phi) sigma_x +
    sin(theta)sin(phi) sigma_y + cos(theta) sigma_z):

    .. math::

        \\gamma = -\\frac{\\Omega}{2}

    where Omega is the solid angle subtended by the path.

    For a circular path at polar angle theta: gamma = -pi(1 - cos(theta)).

    :param H_params: Not used directly; reserved for general Hamiltonians.
    :param path: Parametric path as (n_points, 2) array of (theta, phi).
                 Defaults to circular path at theta=pi/3.
    :param n_points: Number of discretization points.
    :return: DescriptiveResult with Berry phase.
    """
    if path is None:
        theta_0 = np.pi / 3
        phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        path = np.column_stack([np.full(n_points, theta_0), phi])

    thetas = path[:, 0]
    phis = path[:, 1]

    phase = 0.0 + 0j
    for i in range(len(path)):
        j = (i + 1) % len(path)
        theta_i, phi_i = thetas[i], phis[i]
        theta_j, phi_j = thetas[j], phis[j]
        state_i = np.array([np.cos(theta_i / 2), np.sin(theta_i / 2) * np.exp(1j * phi_i)])
        state_j = np.array([np.cos(theta_j / 2), np.sin(theta_j / 2) * np.exp(1j * phi_j)])
        overlap = np.vdot(state_i, state_j)
        phase += np.log(overlap / abs(overlap)) if abs(overlap) > 1e-15 else 0

    berry = -float(np.imag(phase))
    solid_angle = -2 * berry
    analytic = -np.pi * (1 - np.cos(thetas[0]))
    return DescriptiveResult(
        name="berry_phase",
        value=berry,
        extra={
            "berry_phase": berry,
            "solid_angle": solid_angle,
            "analytic_circular": float(analytic),
            "n_points": n_points,
            "theta_0": float(thetas[0]),
        },
    )


def cheatsheet() -> str:
    return "berry_phase(path) -> geometric Berry phase gamma"


berry = berry_phase
