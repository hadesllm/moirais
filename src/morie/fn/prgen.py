# morie.fn — function file (hadesllm/morie)
"""Program enrollment rates."""

from __future__ import annotations

from morie.fn._containers import CrimeResult


def program_enrollment(
    n_enrolled: int,
    n_eligible: int,
) -> CrimeResult:
    """Compute correctional program enrollment rate.

    Parameters
    ----------
    n_enrolled : int
        Number enrolled in program.
    n_eligible : int
        Number eligible for program.

    Returns
    -------
    CrimeResult
    """
    if n_eligible <= 0:
        raise ValueError("n_eligible must be positive")
    rate = n_enrolled / n_eligible
    return CrimeResult(name="program_enrollment", rate=rate, n=n_enrolled, population=n_eligible)


prgen = program_enrollment


def cheatsheet() -> str:
    return "program_enrollment({}) -> Program enrollment rates."
