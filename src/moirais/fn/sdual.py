# moirais.fn — function file (hadesllm/moirais)
"""S-duality transformation g_s <-> 1/g_s."""

from __future__ import annotations

from ._containers import DescriptiveResult


def s_duality(
    g_s: float = 0.1,
) -> DescriptiveResult:
    """Compute S-duality transformed string coupling.

    .. math::

        g_s' = \\frac{1}{g_s}

    S-duality exchanges strong and weak coupling, relating Type I
    to SO(32) heterotic, and Type IIB to itself.

    :param g_s: String coupling constant. Must be > 0.
    :return: DescriptiveResult with dual coupling.
    """
    if g_s <= 0:
        raise ValueError(f"Coupling must be > 0, got {g_s}.")
    g_dual = 1.0 / g_s
    return DescriptiveResult(
        name="s_duality",
        value=g_dual,
        extra={
            "g_s": g_s,
            "g_dual": g_dual,
            "is_self_dual": abs(g_s - 1.0) < 1e-12,
            "regime": "weak" if g_s < 1 else "strong",
            "dual_regime": "strong" if g_s < 1 else "weak",
        },
    )


def cheatsheet() -> str:
    return "s_duality(g_s) -> S-duality g_s <-> 1/g_s"


sdual = s_duality
