# moirais.fn — function file (hadesllm/moirais)
"""Herd immunity threshold."""

from ._containers import ESRes


def herd_immunity_threshold(
    R0: float,
) -> ESRes:
    """Herd immunity threshold.

    .. math::

        HIT = 1 - \\frac{1}{R_0}

    Parameters
    ----------
    R0 : float
        Basic reproduction number. Must be > 1 for meaningful threshold.

    Returns
    -------
    ESRes

    References
    ----------
    Fine, P., Eames, K., & Heymann, D. L. (2011). "Herd immunity": a
    rough guide. Clinical Infectious Diseases, 52(7), 911-916.
    """
    if R0 <= 0:
        raise ValueError("R0 must be positive")
    if R0 <= 1:
        hit = 0.0
    else:
        hit = 1.0 - 1.0 / R0

    return ESRes(
        measure="Herd immunity threshold",
        estimate=float(hit),
        extra={"R0": R0},
    )


herd = herd_immunity_threshold


def cheatsheet() -> str:
    return "herd_immunity_threshold({}) -> Herd immunity threshold."
