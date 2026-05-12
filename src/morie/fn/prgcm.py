# morie.fn -- function file (hadesllm/morie)
"""Program completion rate."""

from __future__ import annotations

from morie.fn._containers import CrimeResult


def program_completion(
    n_completed: int,
    n_started: int,
) -> CrimeResult:
    """Compute correctional program completion rate.

    Parameters
    ----------
    n_completed : int
    n_started : int

    Returns
    -------
    CrimeResult
    """
    if n_started <= 0:
        raise ValueError("n_started must be positive")
    rate = n_completed / n_started
    return CrimeResult(
        name="program_completion",
        rate=rate,
        n=n_completed,
        population=n_started,
        extra={"n_dropout": n_started - n_completed, "dropout_rate": 1 - rate},
    )


prgcm = program_completion


def cheatsheet() -> str:
    return "program_completion({}) -> Program completion rate."
