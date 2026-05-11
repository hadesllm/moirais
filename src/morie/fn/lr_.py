# morie.fn — function file (hadesllm/morie)
"""Likelihood Ratios (LR+, LR-) for diagnostic tests."""

from __future__ import annotations

import math
from typing import Any


def likelihood_ratios(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, Any]:
    """Compute positive and negative likelihood ratios with 95% CIs.

    LR+ = sensitivity / (1 - specificity) = (TP/(TP+FN)) / (FP/(FP+TN))
    LR- = (1 - sensitivity) / specificity = (FN/(TP+FN)) / (TN/(FP+TN))

    CIs use the Simel (1991) log method.

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

    Returns
    -------
    dict
        lr_pos, lr_neg, ci_pos (tuple), ci_neg (tuple).

    References
    ----------
    Deeks, J. J., & Altman, D. G. (2004). Diagnostic tests 4: likelihood
        ratios. *BMJ*, 329(7458), 168-169. doi:10.1136/bmj.329.7458.168
    Simel, D. L., Samsa, G. P., & Matchar, D. B. (1991). Likelihood ratios
        with confidence: sample size estimation for diagnostic test studies.
        *Journal of Clinical Epidemiology*, 44(8), 763-770.
    """
    for name, val in [("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)]:
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}.")

    sens = tp / max(tp + fn, 1)
    spec = tn / max(fp + tn, 1)

    # LR+
    if (1 - spec) > 0:
        lr_pos = sens / (1 - spec)
    else:
        lr_pos = float("inf")

    # LR-
    if spec > 0:
        lr_neg = (1 - sens) / spec
    else:
        lr_neg = float("inf")

    # CIs via log method (Simel 1991)
    ci_pos = _lr_ci(tp, fn, fp, tn, positive=True)
    ci_neg = _lr_ci(tp, fn, fp, tn, positive=False)

    return {
        "lr_pos": lr_pos,
        "lr_neg": lr_neg,
        "ci_pos": ci_pos,
        "ci_neg": ci_neg,
    }


def _lr_ci(
    tp: int,
    fn: int,
    fp: int,
    tn: int,
    *,
    positive: bool,
) -> tuple[float, float]:
    """95% CI for a likelihood ratio via the log method."""
    if positive:
        # LR+ = sens / (1-spec)
        n1 = tp + fn
        n2 = fp + tn
        a = tp
        b = fp
    else:
        # LR- = (1-sens) / spec
        n1 = tp + fn
        n2 = fp + tn
        a = fn
        b = tn

    if a == 0 or b == 0 or n1 == 0 or n2 == 0:
        return (0.0, float("inf"))

    lr_val = (a / n1) / (b / n2)
    # SE of log(LR)
    se_log = math.sqrt(1.0 / a - 1.0 / n1 + 1.0 / b - 1.0 / n2)
    log_lr = math.log(lr_val)

    return (
        math.exp(log_lr - 1.96 * se_log),
        math.exp(log_lr + 1.96 * se_log),
    )


lr_ = likelihood_ratios


def cheatsheet() -> str:
    return "likelihood_ratios({}) -> Likelihood Ratios (LR+, LR-) for diagnostic tests."
