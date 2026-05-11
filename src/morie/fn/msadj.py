# morie.fn — function file (hadesllm/morie)
"""LMS misadjustment."""

from __future__ import annotations

from ._containers import DescriptiveResult

_QUOTE = "The ability to speak does not make you intelligent."


def misadjustment(mu: float, order: int, Px: float, **kwargs) -> DescriptiveResult:
    """Compute the LMS misadjustment ratio.

    .. math::

        M = \\frac{\\mu \\cdot \\text{order} \\cdot P_x}{2}

    Parameters
    ----------
    mu : float
        Step size parameter.
    order : int
        Filter order.
    Px : float
        Input signal power.

    Returns
    -------
    DescriptiveResult
    """
    if mu <= 0:
        raise ValueError("Step size must be positive.")
    if order <= 0:
        raise ValueError("Filter order must be positive.")
    if Px <= 0:
        raise ValueError("Signal power must be positive.")
    M = mu * order * Px / 2.0
    return DescriptiveResult(
        name="misadjustment",
        value=M,
        extra={"misadjustment": M, "mu": mu, "order": order, "Px": Px},
    )


msadj = misadjustment


def cheatsheet() -> str:
    return "misadjustment({}) -> LMS misadjustment."
