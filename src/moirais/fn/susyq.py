"""N-extended SUSY algebra."""

from __future__ import annotations

from ._containers import DescriptiveResult


def susy_algebra(
    N: int = 1,
    d: int = 4,
) -> DescriptiveResult:
    """Compute properties of the N-extended supersymmetry algebra.

    The fundamental anticommutation relation:

    .. math::

        \\{Q_\\alpha^I, \\bar{Q}_{\\dot{\\beta}}^J\\} = 2\\sigma^\\mu_{\\alpha\\dot{\\beta}} P_\\mu \\delta^{IJ}

    :param N: Number of supersymmetries (1, 2, 4, 8).
    :param d: Spacetime dimension.
    :return: DescriptiveResult with algebra properties.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}.")
    spinor_dim = 2 ** (d // 2)
    real_supercharges = N * spinor_dim
    max_spin = 0.5 * N if N <= 4 else 2.0
    multiplet_size = 2**N
    return DescriptiveResult(
        name="susy_algebra",
        value=float(real_supercharges),
        extra={
            "N": N,
            "d": d,
            "real_supercharges": real_supercharges,
            "spinor_dim": spinor_dim,
            "max_spin_in_multiplet": max_spin,
            "multiplet_size": multiplet_size,
            "has_gravity": real_supercharges > 8,
        },
    )


def cheatsheet() -> str:
    return "susy_algebra(N, d) -> N-extended SUSY algebra {Q, Q+} = H"


susyq = susy_algebra
