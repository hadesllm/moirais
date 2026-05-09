# moirais.fn — function file (hadesllm/moirais)
"""Net monetary benefit (NMB)."""

from ._containers import ESRes


def net_monetary_benefit(
    effect_diff: float,
    cost_diff: float,
    wtp: float,
) -> ESRes:
    """Net monetary benefit at given willingness-to-pay threshold.

    .. math::

        NMB = \\lambda \\cdot \\Delta E - \\Delta C

    Parameters
    ----------
    effect_diff : float
    cost_diff : float
    wtp : float
        Willingness-to-pay per QALY.

    Returns
    -------
    ESRes
    """
    nmb = wtp * effect_diff - cost_diff
    cost_effective = nmb > 0

    return ESRes(
        measure="NMB",
        estimate=float(nmb),
        extra={
            "wtp": float(wtp),
            "cost_effective": cost_effective,
            "effect_diff": float(effect_diff),
            "cost_diff": float(cost_diff),
        },
    )


henb = net_monetary_benefit


def cheatsheet() -> str:
    return "net_monetary_benefit({}) -> Net monetary benefit (NMB)."
