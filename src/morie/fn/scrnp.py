# morie.fn — function file (hadesllm/morie)
"""Screening test properties (sensitivity, specificity, PPV, NPV)."""

from __future__ import annotations

import scipy.stats as stats

from ._containers import ESRes


def screening_properties(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    confidence: float = 0.95,
) -> ESRes:
    """Compute screening/diagnostic test properties from a 2x2 table.

    Parameters
    ----------
    tp : int
        True positives.
    fp : int
        False positives.
    fn : int
        False negatives.
    tn : int
        True negatives.
    confidence : float, default 0.95
        Confidence level for Wilson CIs.

    Returns
    -------
    ESRes
        estimate is Youden's J index (sens + spec - 1).

    References
    ----------
    Altman, D. G. & Bland, J. M. (1994). Diagnostic tests 1:
    sensitivity and specificity. BMJ, 308(6943), 1552.
    """
    if any(x < 0 for x in [tp, fp, fn, tn]):
        raise ValueError("All counts must be non-negative")
    if tp + fn == 0 or fp + tn == 0:
        raise ValueError("Need both disease and non-disease subjects")

    sens = tp / (tp + fn)
    spec = tn / (fp + tn)
    ppv = tp / (tp + fp) if tp + fp > 0 else 0.0
    npv = tn / (fn + tn) if fn + tn > 0 else 0.0
    prev = (tp + fn) / (tp + fp + fn + tn)
    lr_pos = sens / (1 - spec) if spec < 1 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("inf")
    youden = sens + spec - 1

    def _wilson_ci(k, n, conf):
        p = k / n
        z = stats.norm.ppf((1 + conf) / 2)
        denom = 1 + z**2 / n
        centre = p + z**2 / (2 * n)
        import numpy as np
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return ((centre - margin) / denom, (centre + margin) / denom)

    sens_ci = _wilson_ci(tp, tp + fn, confidence)
    spec_ci = _wilson_ci(tn, fp + tn, confidence)

    return ESRes(
        measure="screening",
        estimate=float(youden),
        n=tp + fp + fn + tn,
        extra={
            "sensitivity": float(sens),
            "specificity": float(spec),
            "PPV": float(ppv),
            "NPV": float(npv),
            "prevalence": float(prev),
            "LR_positive": float(lr_pos),
            "LR_negative": float(lr_neg),
            "youden_J": float(youden),
            "sens_ci": [float(sens_ci[0]), float(sens_ci[1])],
            "spec_ci": [float(spec_ci[0]), float(spec_ci[1])],
        },
    )


scrnp = screening_properties


def cheatsheet() -> str:
    return "screening_properties({}) -> Screening test sens/spec/PPV/NPV."
