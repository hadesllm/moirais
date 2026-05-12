# morie.fn — function file (hadesllm/morie)
"""Likelihood ratios (positive and negative)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def likelihood_ratios(
    sensitivity: float,
    specificity: float,
) -> DescriptiveResult:
    r"""
    Compute positive and negative likelihood ratios.

    .. math::

        LR^+ = \\frac{\\text{Sens}}{1 - \\text{Spec}}, \\quad
        LR^- = \\frac{1 - \\text{Sens}}{\\text{Spec}}

    Parameters
    ----------
    sensitivity : float
        Test sensitivity (0-1).
    specificity : float
        Test specificity (0-1).

    Returns
    -------
    DescriptiveResult
        extra has 'lr_positive', 'lr_negative'.

    References
    ----------
    Deeks, J. J., & Altman, D. G. (2004). Diagnostic tests 4:
    likelihood ratios. *BMJ*, 329(7458), 168-169.
    """
    if not (0 <= sensitivity <= 1) or not (0 <= specificity <= 1):
        raise ValueError("Sensitivity and specificity must be in [0, 1].")

    lr_pos = sensitivity / (1 - specificity) if specificity < 1 else float("inf")
    lr_neg = (1 - sensitivity) / specificity if specificity > 0 else float("inf")

    return DescriptiveResult(
        name="likelihood_ratios",
        value=lr_pos,
        extra={
            "lr_positive": float(lr_pos),
            "lr_negative": float(lr_neg),
            "sensitivity": sensitivity,
            "specificity": specificity,
        },
    )


lrp = likelihood_ratios


def cheatsheet() -> str:
    return "likelihood_ratios({}) -> Likelihood ratios (positive and negative)."
