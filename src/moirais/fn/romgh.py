# moirais.fn — function file (hadesllm/moirais)
"""Hierarchical omega per subscale."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def omega_hierarchical_sub(
    loadings_general: np.ndarray | list,
    loadings_specific: np.ndarray | list,
) -> ESRes:
    """Hierarchical omega for a subscale from bifactor model.

    omega_h_sub = (sum(lambda_s))^2 / (sum(lambda_s)^2 + sum(lambda_g)^2 + sum(uniqueness))

    Parameters
    ----------
    loadings_general : array-like
        General factor loadings for this subscale's items.
    loadings_specific : array-like
        Specific factor loadings for this subscale's items.

    Returns
    -------
    ESRes
        measure="omega_hierarchical_subscale".

    References
    ----------
    Reise, S. P., Bonifay, W. E., & Haviland, M. G. (2013). Scoring
    and modeling psychological measures in the presence of multidimensionality.
    Journal of Personality Assessment, 95(2), 129-140.
    """
    lg = np.asarray(loadings_general, dtype=np.float64)
    ls = np.asarray(loadings_specific, dtype=np.float64)

    uniqueness = 1 - lg**2 - ls**2
    uniqueness = np.clip(uniqueness, 0, 1)

    total_var = np.sum(lg) ** 2 + np.sum(ls) ** 2 + np.sum(uniqueness)
    omega_hs = np.sum(ls) ** 2 / max(total_var, 1e-10)
    omega_hg = np.sum(lg) ** 2 / max(total_var, 1e-10)

    return ESRes(
        measure="omega_hierarchical_subscale",
        estimate=float(omega_hs),
        extra={
            "omega_h_general": float(omega_hg),
            "k": len(lg),
            "ECV": float(np.sum(lg**2) / max(np.sum(lg**2) + np.sum(ls**2), 1e-10)),
        },
    )


omega_h_sub = omega_hierarchical_sub


def cheatsheet() -> str:
    return "omega_hierarchical_sub({}) -> Hierarchical omega per subscale."
