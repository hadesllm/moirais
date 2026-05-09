# moirais.fn — function file (hadesllm/moirais)
"""Screening test metrics (sensitivity, specificity, PPV, NPV)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def screening_metrics(
    TP: int,
    FP: int,
    FN: int,
    TN: int,
) -> DescriptiveResult:
    """
    Compute screening/diagnostic test performance metrics.

    Parameters
    ----------
    TP : int
        True positives.
    FP : int
        False positives.
    FN : int
        False negatives.
    TN : int
        True negatives.

    Returns
    -------
    DescriptiveResult
        extra has 'sensitivity', 'specificity', 'ppv', 'npv',
        'accuracy', 'prevalence'.

    References
    ----------
    Altman, D. G., & Bland, J. M. (1994). Diagnostic tests 1:
    sensitivity and specificity. *BMJ*, 308(6943), 1552.
    """
    for name, val in [("TP", TP), ("FP", FP), ("FN", FN), ("TN", TN)]:
        if val < 0:
            raise ValueError(f"{name} must be non-negative.")

    total = TP + FP + FN + TN
    if total == 0:
        raise ValueError("Total count must be positive.")

    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    acc = (TP + TN) / total
    prev = (TP + FN) / total

    return DescriptiveResult(
        name="screening_metrics",
        value=acc,
        extra={
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "accuracy": float(acc),
            "prevalence": float(prev),
            "n": total,
        },
    )


scrn = screening_metrics


def cheatsheet() -> str:
    return "screening_metrics({}) -> Screening test metrics (sensitivity, specificity, PPV, NPV)."
