# morie.fn — function file (hadesllm/morie)
"""Instrumental variable Wald estimate (MR-style)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def iv_wald(
    beta_zy: float,
    beta_zx: float,
    *,
    se_zy: float = 0.0,
    se_zx: float = 0.0,
    alpha: float = 0.05,
) -> ESRes:
    r"""
    Wald ratio IV estimator (single instrument).

    .. math::

        \\hat{\\beta}_{IV} = \\frac{\\hat{\\beta}_{ZY}}{\\hat{\\beta}_{ZX}}

    Commonly used in Mendelian randomization with a single genetic
    instrument.

    Parameters
    ----------
    beta_zy : float
        Effect of instrument on outcome.
    beta_zx : float
        Effect of instrument on exposure (first stage).
    se_zy, se_zx : float
        Standard errors (for delta method CI).
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Wald, A. (1940). The fitting of straight lines if both variables
    are subject to error. *Ann Math Stat*, 11(3), 284-300.
    """
    if abs(beta_zx) < 1e-12:
        raise ValueError("beta_zx near zero (weak instrument).")

    beta_iv = beta_zy / beta_zx
    ci_lo, ci_hi, se_iv = None, None, None

    if se_zy > 0 and se_zx > 0:
        se_iv = np.sqrt(se_zy**2 / beta_zx**2 + beta_zy**2 * se_zx**2 / beta_zx**4)
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lo = beta_iv - z * se_iv
        ci_hi = beta_iv + z * se_iv

    return ESRes(
        measure="iv_wald",
        estimate=float(beta_iv),
        ci_lower=float(ci_lo) if ci_lo is not None else None,
        ci_upper=float(ci_hi) if ci_hi is not None else None,
        se=float(se_iv) if se_iv is not None else None,
        extra={"beta_zy": beta_zy, "beta_zx": beta_zx},
    )


ivw = iv_wald


def cheatsheet() -> str:
    return "iv_wald({}) -> Instrumental variable Wald estimate (MR-style)."
