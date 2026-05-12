# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""BPS state bound M >= |Z(charges)|."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bps_state(
    charges: list[float] | np.ndarray | None = None,
    mass: float | None = None,
) -> DescriptiveResult:
    r"""Check the BPS bound for a state with given charges.

    .. math::

        M \\geq |Z| = \\sqrt{\\sum_i Q_i^2}

    BPS states saturate the bound (M = |Z|) and preserve some SUSY.

    :param charges: Electric/magnetic charge vector. Defaults to [1, 0].
    :param mass: Mass of the state. If None, returns the BPS mass.
    :return: DescriptiveResult with BPS bound and saturation check.
    """
    if charges is None:
        charges = [1.0, 0.0]
    charges = np.asarray(charges, dtype=float)
    central_charge_mag = float(np.linalg.norm(charges))
    if mass is None:
        mass = central_charge_mag
    is_bps = abs(mass - central_charge_mag) < 1e-10
    return DescriptiveResult(
        name="bps_state",
        value=central_charge_mag,
        extra={
            "central_charge_magnitude": central_charge_mag,
            "mass": float(mass),
            "is_bps": is_bps,
            "bound_satisfied": mass >= central_charge_mag - 1e-10,
            "charges": charges.tolist(),
        },
    )


def cheatsheet() -> str:
    return "bps_state(charges, mass) -> BPS bound M >= |Z(charges)|"


bpsst = bps_state
