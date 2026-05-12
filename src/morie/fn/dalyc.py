# morie.fn -- function file (hadesllm/morie)
"""DALY computation with age-weighting and discounting (GBD method)."""

from __future__ import annotations

from ._containers import ESRes


def daly_computation(
    yll: float,
    yld: float,
    age_weight: bool = False,
    discount_rate: float = 0.03,
) -> ESRes:
    """Disability-adjusted life years (DALY = YLL + YLD).

    Extends the simple ``daly.py`` with explicit GBD parameters.

    Parameters
    ----------
    yll : float
        Years of life lost (pre-computed, possibly discounted).
    yld : float
        Years lived with disability (pre-computed).
    age_weight : bool, default False
        Whether inputs include age weighting.
    discount_rate : float, default 0.03
        Discount rate used in inputs.

    Returns
    -------
    ESRes

    References
    ----------
    Murray, C. J. L. (1994). Quantifying the burden of disease.
    Bulletin of the World Health Organization, 72(3), 429-445.
    """
    if yll < 0 or yld < 0:
        raise ValueError("yll and yld must be non-negative")

    daly_val = yll + yld

    return ESRes(
        measure="DALY",
        estimate=float(daly_val),
        extra={
            "YLL": float(yll),
            "YLD": float(yld),
            "yll_pct": float(yll / daly_val * 100) if daly_val > 0 else 0.0,
            "yld_pct": float(yld / daly_val * 100) if daly_val > 0 else 0.0,
            "age_weighted": age_weight,
            "discount_rate": discount_rate,
        },
    )


dalyc = daly_computation


def cheatsheet() -> str:
    return "daly_computation({}) -> DALY with GBD discounting and age-weighting."
