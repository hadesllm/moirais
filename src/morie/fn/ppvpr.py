# morie.fn — function file (hadesllm/morie)
"""Knowledge itself is power. — Francis Bacon"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ppv_prevalence(
    sensitivity: float,
    specificity: float,
    prevalence: float,
) -> DescriptiveResult:
    """
    Compute positive and negative predictive values from prevalence.

    .. math::

        PPV = \\frac{Se \\cdot p}{Se \\cdot p + (1 - Sp)(1 - p)}

    .. math::

        NPV = \\frac{Sp (1 - p)}{Sp (1 - p) + (1 - Se) p}

    :param sensitivity: Test sensitivity (Se), proportion in [0, 1].
    :param specificity: Test specificity (Sp), proportion in [0, 1].
    :param prevalence: Disease prevalence (p), proportion in (0, 1).
    :return: DescriptiveResult with PPV value and NPV in extra.
    :raises ValueError: If any parameter is outside valid range.

    References
    ----------
    Altman, D. G., & Bland, J. M. (1994). Diagnostic tests 2: predictive
    values. BMJ, 309(6947), 102. doi:10.1136/bmj.309.6947.102
    """
    for name, val in [("sensitivity", sensitivity), ("specificity", specificity), ("prevalence", prevalence)]:
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {val}.")
    if prevalence == 0.0 or prevalence == 1.0:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}.")

    ppv_val = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
    npv_val = (specificity * (1 - prevalence)) / (specificity * (1 - prevalence) + (1 - sensitivity) * prevalence)

    return DescriptiveResult(
        name="PPV / NPV (prevalence-based)",
        value=float(np.round(ppv_val, 6)),
        extra={
            "ppv": float(np.round(ppv_val, 6)),
            "npv": float(np.round(npv_val, 6)),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "prevalence": prevalence,
        },
    )


ppvpr = ppv_prevalence


def cheatsheet() -> str:
    return "Knowledge itself is power. — Francis Bacon"
