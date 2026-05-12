# morie.fn -- function file (hadesllm/morie)
"""M-theory 11-dimensional supergravity limit."""

from __future__ import annotations

from ._containers import DescriptiveResult


def m_theory_dimension(
    g_s: float = 0.1,
    l_s: float = 1.0,
) -> DescriptiveResult:
    r"""Compute M-theory parameters from string theory data.

    M-theory lives in 11 dimensions. The 11th dimension radius and
    Planck length are:

    .. math::

        R_{11} = g_s \\, l_s, \\quad
        l_P^{(11)} = g_s^{1/3} \\, l_s

    :param g_s: String coupling constant. Must be > 0.
    :param l_s: String length. Must be > 0.
    :return: DescriptiveResult with M-theory parameters.
    """
    if g_s <= 0 or l_s <= 0:
        raise ValueError(f"g_s and l_s must be > 0, got g_s={g_s}, l_s={l_s}.")
    R_11 = g_s * l_s
    l_planck_11 = g_s ** (1.0 / 3.0) * l_s
    return DescriptiveResult(
        name="m_theory_dimension",
        value=11.0,
        extra={
            "spacetime_dimensions": 11,
            "R_11": float(R_11),
            "l_planck_11": float(l_planck_11),
            "g_s": g_s,
            "l_s": l_s,
            "supercharges": 32,
        },
    )


def cheatsheet() -> str:
    return "m_theory_dimension(g_s, l_s) -> M-theory 11D parameters"


mthry = m_theory_dimension
