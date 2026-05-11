# morie.fn — function file (hadesllm/morie)
"""Noise exposure assessment (TWA dBA)."""

import numpy as np

from ._containers import ESRes


def noise_exposure(
    measurements_dba: list | np.ndarray,
    durations: list | np.ndarray,
    oel_dba: float = 85.0,
    criterion_hours: float = 8.0,
) -> ESRes:
    """Compute time-weighted average noise exposure in dBA.

    Uses logarithmic averaging per NIOSH/OSHA standards.

    Parameters
    ----------
    measurements_dba : array-like
        Sound level measurements in dBA.
    durations : array-like
        Duration at each level in hours.
    oel_dba : float
        Occupational exposure limit (default 85 dBA).
    criterion_hours : float

    Returns
    -------
    ESRes
    """
    levels = np.asarray(measurements_dba, dtype=float)
    durs = np.asarray(durations, dtype=float)
    if len(levels) != len(durs):
        raise ValueError("measurements and durations must match")

    total_hours = float(np.sum(durs))
    if total_hours <= 0:
        raise ValueError("Total duration must be positive")

    energy = np.sum(durs * 10 ** (levels / 10))
    twa = float(10 * np.log10(energy / total_hours))

    dose = 0.0
    for l, d in zip(levels, durs):
        allowed = criterion_hours / (2 ** ((l - oel_dba) / 5))
        dose += d / allowed * 100

    return ESRes(
        measure="noise_TWA_dBA",
        estimate=twa,
        extra={"dose_pct": float(dose), "exceeds_oel": twa > oel_dba, "oel_dba": oel_dba, "total_hours": total_hours},
    )


noidx = noise_exposure


def cheatsheet() -> str:
    return "noise_exposure({}) -> Noise exposure assessment (TWA dBA)."
