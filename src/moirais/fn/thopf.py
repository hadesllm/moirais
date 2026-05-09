"""Hopf fibration S^3 -> S^2."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hopf_fibration(n_points: int = 500) -> DescriptiveResult:
    """Compute the Hopf map from S^3 to S^2.

    The Hopf fibration maps (z1, z2) in C^2 with |z1|^2+|z2|^2=1 to:
    :math:`(2\\operatorname{Re}(z_1\\bar{z}_2),\\; 2\\operatorname{Im}(z_1\\bar{z}_2),\\; |z_1|^2 - |z_2|^2)`

    Fibers are great circles in S^3.

    :param n_points: Number of sample points on S^3.
    :return: DescriptiveResult with S^3 and S^2 coordinates.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}.")
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n_points, 4))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    pts = raw / norms
    z1 = pts[:, 0] + 1j * pts[:, 1]
    z2 = pts[:, 2] + 1j * pts[:, 3]
    s2_x = 2.0 * np.real(z1 * np.conj(z2))
    s2_y = 2.0 * np.imag(z1 * np.conj(z2))
    s2_z = np.abs(z1) ** 2 - np.abs(z2) ** 2
    return DescriptiveResult(
        name="hopf_fibration",
        value=float(n_points),
        extra={
            "s3": pts,
            "s2_x": s2_x,
            "s2_y": s2_y,
            "s2_z": s2_z,
            "fiber_type": "S^1",
        },
    )


def cheatsheet() -> str:
    return "hopf_fibration(n_points) -> Hopf map S^3 -> S^2"
