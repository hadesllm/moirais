# morie.fn -- function file (hadesllm/morie)
"""Disability-Adjusted Life Years (DALY)."""

from __future__ import annotations

from typing import Any


def disability_adjusted_life_years(
    yll: float,
    yld: float,
) -> dict[str, Any]:
    """
    Compute Disability-Adjusted Life Years (DALY = YLL + YLD).

    .. math::

        \\text{DALY} = \\text{YLL} + \\text{YLD}

    * YLL = Years of Life Lost (premature mortality).
    * YLD = Years Lived with Disability (morbidity).

    :param yll: Years of life lost.
    :param yld: Years lived with disability.
    :return: Dictionary with daly, yll, yld.
    :raises ValueError: If yll or yld is negative.

    References
    ----------
    Murray, C. J. L., & Lopez, A. D. (1996). *The Global Burden of
    Disease*. Harvard School of Public Health / WHO.
    """
    if yll < 0:
        raise ValueError("yll must be non-negative.")
    if yld < 0:
        raise ValueError("yld must be non-negative.")

    return {
        "daly": float(yll + yld),
        "yll": float(yll),
        "yld": float(yld),
    }


daly = disability_adjusted_life_years


def cheatsheet() -> str:
    return "disability_adjusted_life_years({}) -> Disability-Adjusted Life Years (DALY)."
