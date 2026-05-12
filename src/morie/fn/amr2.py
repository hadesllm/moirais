# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""A-M R-squared goodness of fit."""

from __future__ import annotations

from ._containers import DescriptiveResult


def am_r_squared(actual_positions, estimated) -> DescriptiveResult:
    """R-squared between actual and A-M estimated positions.

    :param actual_positions: True/benchmark positions.
    :param estimated: A-M estimated positions.
    :return: DescriptiveResult with R-squared.

    .. epigraph:: "People die when they are killed." -- Shirou Emiya, Fate/stay night
    """
    import numpy as np

    y = np.asarray(actual_positions, dtype=float).ravel()
    yhat = np.asarray(estimated, dtype=float).ravel()
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="am_r_squared",
        value=float(r2),
        extra={"ss_residual": float(ss_res), "ss_total": float(ss_tot)},
    )


amr2 = am_r_squared


def cheatsheet() -> str:
    return "am_r_squared({}) -> A-M R-squared goodness of fit."
