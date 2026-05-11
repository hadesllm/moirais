# morie.fn — function file (hadesllm/morie)
"""Standard error of measurement. 'Patience you must have.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def sem_measurement(
    sd: float,
    reliability: float,
) -> DescriptiveResult:
    """
    Standard error of measurement (SEM).

    .. math::

        SEM = SD \\sqrt{1 - \\rho_{xx'}}

    :param sd: Standard deviation of the observed scores.
    :param reliability: Reliability coefficient (e.g. alpha), in (0, 1).
    :return: DescriptiveResult with SEM as value.
    :raises ValueError: If sd <= 0 or reliability not in (0, 1).

    References
    ----------
    Harvill, L. M. (1991). Standard error of measurement. Educational
    Measurement: Issues and Practice, 10(2), 33--41.
    doi:10.1111/j.1745-3992.1991.tb00195.x
    """
    if sd <= 0.0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    if not 0.0 < reliability < 1.0:
        raise ValueError(f"reliability must be in (0, 1), got {reliability}.")

    sem = sd * np.sqrt(1.0 - reliability)

    return DescriptiveResult(
        name="Standard Error of Measurement",
        value=float(np.round(sem, 4)),
        extra={
            "sem": float(np.round(sem, 4)),
            "sd": sd,
            "reliability": reliability,
        },
    )


semea = sem_measurement


def cheatsheet() -> str:
    return "sem_measurement({}) -> Standard error of measurement. 'Patience you must have.' -- "
