# morie.fn -- function file (hadesllm/morie)
"""Diversion program utilization."""

from __future__ import annotations

from morie.fn._containers import CrimeResult


def court_diversion(
    n_diverted: int,
    n_eligible: int,
) -> CrimeResult:
    """Compute diversion program utilization rate.

    Parameters
    ----------
    n_diverted : int
        Number diverted from traditional prosecution.
    n_eligible : int
        Number eligible for diversion.

    Returns
    -------
    CrimeResult
    """
    if n_eligible <= 0:
        raise ValueError("n_eligible must be positive")
    rate = n_diverted / n_eligible
    return CrimeResult(
        name="diversion_rate",
        rate=rate,
        n=n_diverted,
        population=n_eligible,
        extra={"n_not_diverted": n_eligible - n_diverted},
    )


crtdv = court_diversion


def cheatsheet() -> str:
    return "court_diversion({}) -> Diversion program utilization."
