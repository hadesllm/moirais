# morie.fn — function file (hadesllm/morie)
"""GKP-Witten bulk-to-boundary propagator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gkp_witten(
    phi_boundary: float = 1.0,
    z_bulk: float | np.ndarray = 0.5,
    delta: float = 3.0,
    d: int = 4,
    L: float = 1.0,
) -> DescriptiveResult:
    r"""Compute the GKP-Witten bulk-to-boundary propagator in AdS.

    For a scalar field with conformal dimension Delta in AdS_{d+1}:

    .. math::

        K_\\Delta(z, x; x') \\propto
        \\left(\\frac{z}{z^2 + |x - x'|^2}\\right)^\\Delta

    Evaluated at x = x' (coincident point):

    .. math::

        \\phi(z) = \\phi_0 \\, z^{d - \\Delta}

    :param phi_boundary: Boundary value of the field.
    :param z_bulk: Bulk radial coordinate(s). Must be > 0.
    :param delta: Conformal dimension.
    :param d: Boundary dimension.
    :param L: AdS radius.
    :return: DescriptiveResult with bulk field profile.
    """
    z = np.atleast_1d(np.asarray(z_bulk, dtype=float))
    if np.any(z <= 0):
        raise ValueError("z_bulk must be > 0.")
    phi_bulk = phi_boundary * (z / L) ** (d - delta)
    normalizable = phi_boundary * (z / L) ** delta
    return DescriptiveResult(
        name="gkp_witten",
        value=float(phi_bulk[0]) if phi_bulk.size == 1 else None,
        extra={
            "phi_bulk": phi_bulk,
            "normalizable_mode": normalizable,
            "delta": delta,
            "d": d,
            "L": L,
            "z_bulk": z,
        },
    )


def cheatsheet() -> str:
    return "gkp_witten(phi_boundary, z_bulk, delta) -> bulk-to-boundary propagator"


gkpw = gkp_witten
