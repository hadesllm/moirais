# morie.fn — function file (hadesllm/morie)
"""MDS goodness-of-fit interpretation. 'Infinite Void.' -- Gojo, Jujutsu Kaisen"""

from __future__ import annotations

from ._containers import DescriptiveResult


def gof_mds_interpret(stress):
    """Interpret Kruskal stress value as quality label.

    Parameters
    ----------
    stress : float
        Kruskal stress-1 value.

    Returns
    -------
    DescriptiveResult
        value = quality label (str), extra has stress and thresholds.

    Notes
    -----
    Kruskal (1964) thresholds: <0.025 excellent, <0.05 good, <0.10 fair, <0.20 poor, >=0.20 unacceptable.
    """
    stress = float(stress)
    if stress < 0.025:
        label = "excellent"
    elif stress < 0.05:
        label = "good"
    elif stress < 0.10:
        label = "fair"
    elif stress < 0.20:
        label = "poor"
    else:
        label = "unacceptable"
    return DescriptiveResult(
        name="gof_mds_interpret",
        value=label,
        extra={"stress": stress, "thresholds": {"excellent": 0.025, "good": 0.05, "fair": 0.10, "poor": 0.20}},
    )


gofmd = gof_mds_interpret


def cheatsheet() -> str:
    return "gof_mds_interpret({}) -> MDS goodness-of-fit interpretation. 'Infinite Void.' -- Gojo"
