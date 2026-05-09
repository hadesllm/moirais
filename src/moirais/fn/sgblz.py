"""Boltzmann acceptance criterion for simulated annealing."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def boltzmann_accept(
    energy_old: float,
    energy_new: float,
    temperature: float,
) -> SpatialResult:
    r"""Boltzmann acceptance probability.

    .. math::

        P(\text{accept}) = \min\!\bigl(1, \exp(-\Delta E / T)\bigr)

    Parameters
    ----------
    energy_old : float
        Current energy.
    energy_new : float
        Proposed energy.
    temperature : float
        Current temperature (> 0).

    Returns
    -------
    SpatialResult
        ``statistic`` is the acceptance probability.
        ``extra`` has ``delta_energy``, ``accept``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.

    .. epigraph::

        "Someday the light will return." -- FF15
    """
    dE = energy_new - energy_old
    if temperature <= 0:
        prob = 1.0 if dE <= 0 else 0.0
    elif dE <= 0:
        prob = 1.0
    else:
        prob = float(np.exp(-dE / temperature))

    return SpatialResult(
        name="boltzmann_accept",
        statistic=prob,
        p_value=None,
        extra={"delta_energy": dE, "accept": prob >= 0.5},
    )


sgblz = boltzmann_accept


def cheatsheet() -> str:
    return "boltzmann_accept({}) -> Boltzmann acceptance criterion for simulated annealing."
