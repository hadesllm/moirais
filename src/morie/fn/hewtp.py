# morie.fn -- function file (hadesllm/morie)
"""Willingness-to-pay threshold analysis."""

from ._containers import ESRes


def willingness_to_pay(
    effect: float,
    cost: float,
    wtp_per_qaly: float = 50000.0,
) -> ESRes:
    """Determine if intervention is cost-effective at WTP threshold.

    Parameters
    ----------
    effect : float
        Incremental QALYs gained.
    cost : float
        Incremental cost.
    wtp_per_qaly : float
        WTP threshold per QALY (default $50,000).

    Returns
    -------
    ESRes
    """
    icer = cost / effect if effect != 0 else float("inf")
    cost_effective = icer <= wtp_per_qaly if effect > 0 else False
    nmb = wtp_per_qaly * effect - cost

    return ESRes(
        measure="WTP_analysis",
        estimate=float(icer),
        extra={"cost_effective": cost_effective, "nmb": float(nmb), "wtp_threshold": float(wtp_per_qaly)},
    )


hewtp = willingness_to_pay


def cheatsheet() -> str:
    return "willingness_to_pay({}) -> Willingness-to-pay threshold analysis."
