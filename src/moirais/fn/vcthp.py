"""Help-seeking behavior rates."""

from __future__ import annotations

from moirais.fn._containers import CrimeResult


def victim_help_seeking(
    sought_help: int,
    total_victims: int,
) -> CrimeResult:
    """Compute victim help-seeking rate.

    Parameters
    ----------
    sought_help : int
        Number of victims who sought help.
    total_victims : int
        Total victims.

    Returns
    -------
    CrimeResult
    """
    if total_victims <= 0:
        raise ValueError("total_victims must be positive")
    rate = sought_help / total_victims
    return CrimeResult(
        name="help_seeking_rate",
        rate=rate,
        n=sought_help,
        population=total_victims,
    )


vcthp = victim_help_seeking


def cheatsheet() -> str:
    return "victim_help_seeking({}) -> Help-seeking behavior rates."
