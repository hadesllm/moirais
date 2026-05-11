# morie.fn — function file (hadesllm/morie)
"""Chern-Simons level and partition function."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def chern_simons(
    k: int = 1,
    gauge_group: str = "SU(2)",
    manifold: str = "S3",
) -> DescriptiveResult:
    """Compute Chern-Simons theory properties.

    For SU(2) Chern-Simons on S^3:

    .. math::

        Z_{CS}(S^3) = \\sqrt{\\frac{2}{k+2}}
                      \\sin\\!\\left(\\frac{\\pi}{k+2}\\right)

    :param k: Chern-Simons level (positive integer).
    :param gauge_group: Gauge group name (for metadata).
    :param manifold: 3-manifold (S3 has exact formula).
    :return: DescriptiveResult with partition function value.
    """
    if k < 1:
        raise ValueError(f"CS level must be >= 1, got {k}.")
    if manifold == "S3" and gauge_group == "SU(2)":
        Z = np.sqrt(2.0 / (k + 2)) * np.sin(np.pi / (k + 2))
    else:
        Z = np.sqrt(2.0 / (k + 2))
    central_charge_cs = 3 * k / (k + 2)
    return DescriptiveResult(
        name="chern_simons",
        value=float(Z),
        extra={
            "partition_function": float(Z),
            "k": k,
            "gauge_group": gauge_group,
            "manifold": manifold,
            "central_charge": float(central_charge_cs),
        },
    )


def cheatsheet() -> str:
    return "chern_simons(k, gauge_group) -> CS level and partition function"


chsim = chern_simons
