"""Vaccine efficacy with Greenwood and exact confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st


def vaccine_efficacy_exact(
    cases_vacc: int,
    n_vacc: int,
    cases_ctrl: int,
    n_ctrl: int,
    *,
    person_time_vacc: float | None = None,
    person_time_ctrl: float | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Vaccine efficacy with Greenwood and exact (Clopper-Pearson) CIs.

    VE = 1 - RR. When person-time is provided, uses incidence rates;
    otherwise uses cumulative incidence (attack rates).

    .. math::

        VE = 1 - \\frac{\\text{AR}_{\\text{vacc}}}{\\text{AR}_{\\text{ctrl}}}

    Parameters
    ----------
    cases_vacc : int
        Cases in vaccinated group.
    n_vacc : int
        Size of vaccinated group.
    cases_ctrl : int
        Cases in control/unvaccinated group.
    n_ctrl : int
        Size of control group.
    person_time_vacc : float or None
        Person-time at risk for vaccinated (for rate-based VE).
    person_time_ctrl : float or None
        Person-time at risk for control.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: 've', 'ci_lower_greenwood', 'ci_upper_greenwood',
              'ci_lower_exact', 'ci_upper_exact', 'relative_risk'.

    References
    ----------
    Orenstein, W. A. et al. (1985). Field evaluation of vaccine
    efficacy. Bulletin of the WHO, 63(6), 1055-1068.
    """
    if n_vacc <= 0 or n_ctrl <= 0:
        raise ValueError("Group sizes must be positive.")
    if cases_vacc < 0 or cases_ctrl < 0:
        raise ValueError("Case counts must be non-negative.")

    if person_time_vacc is not None and person_time_ctrl is not None:
        rate_v = cases_vacc / person_time_vacc
        rate_c = cases_ctrl / person_time_ctrl
        rr = rate_v / rate_c if rate_c > 0 else np.inf
    else:
        ar_v = cases_vacc / n_vacc
        ar_c = cases_ctrl / n_ctrl
        rr = ar_v / ar_c if ar_c > 0 else np.inf

    ve = (1 - rr) * 100

    se_log_rr = np.sqrt(
        1 / max(cases_vacc, 0.5) - 1 / n_vacc
        + 1 / max(cases_ctrl, 0.5) - 1 / n_ctrl
    )
    z = _st.norm.ppf(1 - alpha / 2)
    rr_lo_g = rr * np.exp(-z * se_log_rr)
    rr_hi_g = rr * np.exp(z * se_log_rr)
    ve_lo_g = (1 - rr_hi_g) * 100
    ve_hi_g = (1 - rr_lo_g) * 100

    total_cases = cases_vacc + cases_ctrl
    if total_cases > 0:
        p = cases_vacc / total_cases
        ci_lo_p = _st.beta.ppf(alpha / 2, cases_vacc + 0.5, cases_ctrl + 0.5) if cases_vacc > 0 else 0.0
        ci_hi_p = _st.beta.ppf(1 - alpha / 2, cases_vacc + 0.5, cases_ctrl + 0.5) if cases_vacc < total_cases else 1.0
        rr_lo_e = (ci_lo_p / (1 - ci_lo_p)) * (n_ctrl / n_vacc)
        rr_hi_e = (ci_hi_p / (1 - ci_hi_p)) * (n_ctrl / n_vacc)
        ve_lo_e = (1 - rr_hi_e) * 100
        ve_hi_e = (1 - rr_lo_e) * 100
    else:
        ve_lo_e = np.nan
        ve_hi_e = np.nan

    return {
        "ve": float(ve),
        "ci_lower_greenwood": float(ve_lo_g),
        "ci_upper_greenwood": float(ve_hi_g),
        "ci_lower_exact": float(ve_lo_e),
        "ci_upper_exact": float(ve_hi_e),
        "relative_risk": float(rr),
    }


vaces = vaccine_efficacy_exact


def cheatsheet() -> str:
    return "vaccine_efficacy_exact({}) -> VE with Greenwood and exact CI."
