# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""BIC-based Bayes factor approximation."""

from __future__ import annotations

__all__ = ["bayes_factor_bic", "bic2b"]

import math
from typing import Any


def bayes_factor_bic(
    bic_null: float,
    bic_alt: float,
) -> dict[str, Any]:
    """
    Approximate Bayes factor from BIC values of two competing models.

    .. math::

        BF_{01} \\approx \\exp\\bigl((BIC_1 - BIC_0) / 2\\bigr)

    where model 0 is the null (simpler) and model 1 is the alternative.

    Evidence categories follow Kass & Raftery (1995):
      * BF > 150: Very strong for null
      * BF 20--150: Strong for null
      * BF 3--20: Positive for null
      * BF 1--3: Barely worth mentioning (null)
      * BF 1/3--1: Barely worth mentioning (alt)
      * BF 1/20--1/3: Positive for alt
      * BF 1/150--1/20: Strong for alt
      * BF < 1/150: Very strong for alt

    Parameters
    ----------
    bic_null : float
        BIC of the null (simpler) model.
    bic_alt : float
        BIC of the alternative model.

    Returns
    -------
    dict
        bf01, bf10, evidence_category, delta_bic

    References
    ----------
    Kass, R. E. & Raftery, A. E. (1995). Bayes factors. *JASA*,
    90(430), 773--795.
    Wagenmakers, E.-J. (2007). *Psychonomic Bulletin & Review*,
    14(5), 779--804.
    """
    diff = bic_alt - bic_null
    diff_clamped = max(min(diff, 700), -700)
    bf01 = math.exp(diff_clamped / 2.0)
    bf10 = 1.0 / bf01 if bf01 > 0 else float("inf")

    if bf01 > 150:
        cat = "Very strong evidence for null"
    elif bf01 > 20:
        cat = "Strong evidence for null"
    elif bf01 > 3:
        cat = "Positive evidence for null"
    elif bf01 > 1:
        cat = "Barely worth mentioning (null)"
    elif bf01 > 1.0 / 3.0:
        cat = "Barely worth mentioning (alt)"
    elif bf01 > 1.0 / 20.0:
        cat = "Positive evidence for alternative"
    elif bf01 > 1.0 / 150.0:
        cat = "Strong evidence for alternative"
    else:
        cat = "Very strong evidence for alternative"

    return {
        "bf01": bf01,
        "bf10": bf10,
        "evidence_category": cat,
        "delta_bic": float(diff),
    }


bic2b = bayes_factor_bic


def cheatsheet() -> str:
    return "bayes_factor_bic(bic_null, bic_alt) -> BIC-based Bayes factor approximation."
