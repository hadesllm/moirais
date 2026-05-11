# morie.fn — function file (hadesllm/morie)
"""Cost savings from correctional program."""

from __future__ import annotations

from morie.fn._containers import ESRes


def program_cost_savings(
    effect_size: float,
    cost_per_recid: float,
    n_participants: int,
    program_cost: float,
) -> ESRes:
    """Estimate cost savings from avoided recidivism.

    Parameters
    ----------
    effect_size : float
        Absolute risk reduction in recidivism (0 to 1).
    cost_per_recid : float
        Cost per recidivism event (incarceration, courts, etc.).
    n_participants : int
        Number of program participants.
    program_cost : float
        Total program cost.

    Returns
    -------
    ESRes
    """
    if n_participants <= 0:
        raise ValueError("n_participants must be positive")
    avoided = abs(effect_size) * n_participants
    savings = avoided * cost_per_recid
    net = savings - program_cost
    roi = net / program_cost if program_cost > 0 else float("inf")
    return ESRes(
        measure="program_cost_savings",
        estimate=net,
        n=n_participants,
        extra={"avoided_recid": avoided, "gross_savings": savings, "program_cost": program_cost, "roi": roi},
    )


prgcs = program_cost_savings


def cheatsheet() -> str:
    return "program_cost_savings({}) -> Cost savings from correctional program."
