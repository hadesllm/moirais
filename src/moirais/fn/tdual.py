"""T-duality transformation R <-> alpha'/R."""

from __future__ import annotations

from ._containers import DescriptiveResult


def t_duality(
    R: float = 1.0,
    alpha_prime: float = 1.0,
) -> DescriptiveResult:
    """Compute T-duality transformed radius.

    .. math::

        R' = \\frac{\\alpha'}{R}

    T-duality exchanges momentum and winding modes, mapping a string
    on a circle of radius R to one of radius alpha'/R.

    :param R: Original compactification radius. Must be > 0.
    :param alpha_prime: String length squared. Must be > 0.
    :return: DescriptiveResult with dual radius.
    """
    if R <= 0:
        raise ValueError(f"Radius must be > 0, got {R}.")
    if alpha_prime <= 0:
        raise ValueError(f"alpha_prime must be > 0, got {alpha_prime}.")
    R_dual = alpha_prime / R
    self_dual = R**2  # self-dual radius at R = sqrt(alpha')
    return DescriptiveResult(
        name="t_duality",
        value=R_dual,
        extra={
            "R": R,
            "R_dual": R_dual,
            "alpha_prime": alpha_prime,
            "self_dual_radius": float(alpha_prime**0.5),
            "is_self_dual": abs(R - alpha_prime**0.5) < 1e-12,
        },
    )


def cheatsheet() -> str:
    return "t_duality(R, alpha_prime) -> T-duality R <-> alpha'/R"


tdual = t_duality
