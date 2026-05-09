# moirais.fn — function file (hadesllm/moirais)
"""Pre-post program effect size (Cohen's d)."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def program_effect(
    pre_scores: np.ndarray | list[float],
    post_scores: np.ndarray | list[float],
) -> ESRes:
    """Compute pre-post program effect size (Cohen's d for paired data).

    Parameters
    ----------
    pre_scores : array-like
    post_scores : array-like

    Returns
    -------
    ESRes
    """
    pre = np.asarray(pre_scores, dtype=float)
    post = np.asarray(post_scores, dtype=float)
    if len(pre) != len(post) or len(pre) < 2:
        raise ValueError("pre and post must be same length >= 2")
    diff = post - pre
    d = float(np.mean(diff) / np.std(diff, ddof=1))
    se = np.sqrt(1 / len(diff) + d**2 / (2 * len(diff)))
    ci_lo = d - 1.96 * se
    ci_hi = d + 1.96 * se
    return ESRes(
        measure="cohens_d_paired",
        estimate=d,
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        se=float(se),
        n=len(pre),
        extra={"mean_diff": float(np.mean(diff)), "sd_diff": float(np.std(diff, ddof=1))},
    )


prgef = program_effect


def cheatsheet() -> str:
    return "program_effect({}) -> Pre-post program effect size (Cohen's d)."
