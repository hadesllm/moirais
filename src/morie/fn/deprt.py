# morie.fn — function file (hadesllm/morie)
"""Age dependency ratio."""

from ._containers import ESRes


def dependency_ratio(
    pop_0_14: int,
    pop_15_64: int,
    pop_65plus: int,
) -> ESRes:
    """Compute age dependency ratios.

    Parameters
    ----------
    pop_0_14 : int
    pop_15_64 : int
    pop_65plus : int

    Returns
    -------
    ESRes
    """
    if pop_15_64 <= 0:
        raise ValueError("Working-age population must be positive")

    youth = pop_0_14 / pop_15_64 * 100
    old_age = pop_65plus / pop_15_64 * 100
    total = (pop_0_14 + pop_65plus) / pop_15_64 * 100

    return ESRes(
        measure="dependency_ratio",
        estimate=float(total),
        extra={
            "youth_ratio": float(youth),
            "old_age_ratio": float(old_age),
            "total_population": pop_0_14 + pop_15_64 + pop_65plus,
        },
    )


deprt = dependency_ratio


def cheatsheet() -> str:
    return "dependency_ratio({}) -> Age dependency ratio."
