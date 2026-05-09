# moirais.fn — function file (hadesllm/moirais)
"""Etiologic fraction among the exposed."""

from __future__ import annotations

from ._containers import ESRes


def etiologic_fraction(
    RR: float,
) -> ESRes:
    """
    Compute the etiologic fraction (attributable fraction among exposed).

    .. math::

        EF = \\frac{RR - 1}{RR}

    Parameters
    ----------
    RR : float
        Relative risk.

    Returns
    -------
    ESRes

    References
    ----------
    Miettinen, O. S. (1974). Proportion of disease caused or prevented
    by a given exposure, trait or intervention. *Am J Epidemiol*,
    99(5), 325-332.
    """
    if RR <= 0:
        raise ValueError("RR must be positive.")

    ef = (RR - 1) / RR

    return ESRes(
        measure="etiologic_fraction",
        estimate=float(ef),
        extra={"RR": RR},
    )


etiof = etiologic_fraction


def cheatsheet() -> str:
    return "etiologic_fraction({}) -> Etiologic fraction among the exposed."
