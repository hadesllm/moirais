"""Modular parameter of a complex torus."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_modular(tau: complex = 1j) -> DescriptiveResult:
    """Compute properties of the complex torus C / (Z + tau*Z).

    The modular parameter *tau* must have positive imaginary part.
    The torus has area Im(tau) (for the lattice Z + tau*Z).

    :param tau: Modular parameter (Im(tau) > 0).
    :return: DescriptiveResult with lattice info in *extra*.
    :raises ValueError: If Im(tau) <= 0.
    """
    if np.imag(tau) <= 0:
        raise ValueError(f"Im(tau) must be > 0, got Im(tau)={np.imag(tau)}.")
    area = float(np.imag(tau))
    a = np.real(tau)
    b = np.imag(tau)
    modulus = float(np.abs(tau))
    in_fundamental_domain = (-0.5 <= a < 0.5) and (modulus >= 1.0)
    return DescriptiveResult(
        name="torus_modular",
        value=area,
        extra={
            "tau": tau,
            "area": area,
            "real_part": float(a),
            "imag_part": float(b),
            "modulus": modulus,
            "in_fundamental_domain": in_fundamental_domain,
        },
    )


def cheatsheet() -> str:
    return "torus_modular(tau) -> modular parameter of complex torus"
