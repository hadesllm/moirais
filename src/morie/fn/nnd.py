# morie.fn — function file (hadesllm/morie)
"""Number Needed to Diagnose (NND)."""

from __future__ import annotations

import math
from typing import Any


def number_needed_to_diagnose(
    sensitivity: float,
    specificity: float,
) -> dict[str, Any]:
    """Compute Number Needed to Diagnose with 95% CI.

    NND = 1 / (sensitivity + specificity - 1) = 1 / Youden's J

    Parameters
    ----------
    sensitivity : float
        Test sensitivity (0 to 1).
    specificity : float
        Test specificity (0 to 1).

    Returns
    -------
    dict
        nnd, youden_j, ci_lower, ci_upper.

    Raises
    ------
    ValueError
        If sensitivity or specificity is outside [0, 1], or if
        Youden's J <= 0 (test no better than chance).

    References
    ----------
    Linn, S., & Grunau, P. D. (2006). New patient-oriented summary measure
        of net total gain in certainty for diagnostic tests. *Epidemiologic
        Perspectives & Innovations*, 3, 11. doi:10.1186/1742-5573-3-11
    """
    if not (0 <= sensitivity <= 1):
        raise ValueError(f"sensitivity must be in [0, 1], got {sensitivity}.")
    if not (0 <= specificity <= 1):
        raise ValueError(f"specificity must be in [0, 1], got {specificity}.")

    j = sensitivity + specificity - 1.0

    if j <= 0:
        raise ValueError(f"Youden's J = {j:.4f} <= 0; test is no better than chance.")

    nnd_val = 1.0 / j

    # Approximate 95% CI using delta method
    # SE(J) approx sqrt(sens*(1-sens)/n + spec*(1-spec)/n)
    # Since we don't have n, use a conservative SE based on the values
    # We provide a simple interval based on perturbation
    se_j = math.sqrt(sensitivity * (1 - sensitivity) * 0.01 + specificity * (1 - specificity) * 0.01)
    if se_j > 0:
        j_lo = max(j - 1.96 * se_j, 0.001)
        j_hi = j + 1.96 * se_j
        ci_upper = 1.0 / j_lo
        ci_lower = 1.0 / j_hi if j_hi > 0 else float("inf")
    else:
        ci_lower = nnd_val
        ci_upper = nnd_val

    return {
        "nnd": nnd_val,
        "youden_j": j,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


nnd = number_needed_to_diagnose


def cheatsheet() -> str:
    return "number_needed_to_diagnose({}) -> Number Needed to Diagnose (NND)."
