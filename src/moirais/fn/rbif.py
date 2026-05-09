# moirais.fn — function file (hadesllm/moirais)
"""Omega from bifactor model."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def bifactor_omega(
    general_loadings: np.ndarray | list,
    specific_loadings: np.ndarray | list,
    error_var: np.ndarray | list,
) -> ESRes:
    """Omega total and omega hierarchical from a bifactor model.

    omega_total = 1 - sum(error) / total_variance
    omega_h = sum(lambda_g)^2 / total_variance

    Parameters
    ----------
    general_loadings : array-like
        General factor loadings (p,).
    specific_loadings : array-like
        Specific factor loadings (p,).
    error_var : array-like
        Error/uniqueness variance (p,).

    Returns
    -------
    ESRes
        measure="bifactor_omega".

    References
    ----------
    Reise, S. P. (2012). The rediscovery of bifactor measurement models.
    Multivariate Behavioral Research, 47(5), 667-696.
    """
    lg = np.asarray(general_loadings, dtype=np.float64)
    ls = np.asarray(specific_loadings, dtype=np.float64)
    ev = np.asarray(error_var, dtype=np.float64)

    total_var = np.sum(lg) ** 2 + np.sum(ls) ** 2 + np.sum(ev)
    omega_total = 1 - np.sum(ev) / max(total_var, 1e-10)
    omega_h = np.sum(lg) ** 2 / max(total_var, 1e-10)
    ecv = np.sum(lg**2) / max(np.sum(lg**2) + np.sum(ls**2), 1e-10)

    return ESRes(
        measure="bifactor_omega",
        estimate=float(omega_total),
        extra={
            "omega_h": float(omega_h),
            "ECV": float(ecv),
            "k": len(lg),
        },
    )


bif_omega = bifactor_omega


def cheatsheet() -> str:
    return "bifactor_omega({}) -> Omega from bifactor model."
