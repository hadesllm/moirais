# morie.fn -- function file (hadesllm/morie)
"""Cost-effectiveness acceptability curve (CEAC)."""

import numpy as np

from ._containers import DescriptiveResult


def ceac(
    cost_diffs: list | np.ndarray,
    effect_diffs: list | np.ndarray,
    wtp_range: list | np.ndarray | None = None,
) -> DescriptiveResult:
    """Compute CEAC: probability of cost-effectiveness at each WTP.

    Parameters
    ----------
    cost_diffs : array-like
    effect_diffs : array-like
    wtp_range : array-like or None
        WTP thresholds to evaluate. Default: 0 to 150000 by 5000.

    Returns
    -------
    DescriptiveResult
    """
    c = np.asarray(cost_diffs, dtype=float)
    e = np.asarray(effect_diffs, dtype=float)
    if len(c) != len(e):
        raise ValueError("Lengths must match")

    if wtp_range is None:
        wtp_range = np.arange(0, 155000, 5000)
    wtp = np.asarray(wtp_range, dtype=float)

    probs = []
    for w in wtp:
        nmb = w * e - c
        probs.append(float(np.mean(nmb > 0)))

    return DescriptiveResult(
        name="CEAC",
        value={"wtp": wtp.tolist(), "probability": probs},
        extra={"n_simulations": len(c), "n_wtp_points": len(wtp)},
    )


heceac = ceac


def cheatsheet() -> str:
    return "ceac({}) -> Cost-effectiveness acceptability curve (CEAC)."
