# morie.fn -- function file (hadesllm/morie)
"""Case fatality rate (CFR) with interval censoring correction."""

from __future__ import annotations

from typing import Any

from scipy import stats as _st


def case_fatality_rate(
    deaths: int,
    cases: int,
    *,
    resolved: int | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute case fatality rate with optional interval-censoring correction.

    The naive CFR = deaths / cases is biased during an ongoing epidemic
    because cases with unknown outcomes are in the denominator. The
    corrected CFR uses only resolved cases in the denominator.

    .. math::

        CFR_{\\text{naive}} = \\frac{D}{C}

        CFR_{\\text{corrected}} = \\frac{D}{D + R}

    where D = deaths, C = total cases, R = recovered/resolved.

    Parameters
    ----------
    deaths : int
        Number of deaths.
    cases : int
        Total confirmed cases.
    resolved : int or None
        Number of resolved cases (deaths + recovered). If None,
        only the naive CFR is returned.
    alpha : float, default 0.05
        Significance level for exact binomial (Clopper-Pearson) CI.

    Returns
    -------
    dict
        Keys: 'cfr_naive', 'cfr_corrected' (if resolved given),
              'ci_lower', 'ci_upper'.

    References
    ----------
    Nishiura, H. et al. (2009). Early epidemiological assessment of
    the virulence of emerging infectious diseases: a case study of an
    influenza pandemic. PLoS ONE, 4(8), e6852.
    """
    if deaths < 0 or cases <= 0:
        raise ValueError("deaths >= 0 and cases > 0 required.")
    if deaths > cases:
        raise ValueError("deaths cannot exceed cases.")

    cfr_naive = deaths / cases

    ci_lo_naive = _st.beta.ppf(alpha / 2, deaths, cases - deaths + 1) if deaths > 0 else 0.0
    ci_hi_naive = _st.beta.ppf(1 - alpha / 2, deaths + 1, cases - deaths) if deaths < cases else 1.0

    result = {
        "cfr_naive": float(cfr_naive),
        "ci_lower": float(ci_lo_naive),
        "ci_upper": float(ci_hi_naive),
        "deaths": deaths,
        "cases": cases,
    }

    if resolved is not None:
        if resolved < deaths:
            raise ValueError("resolved must be >= deaths.")
        if resolved > cases:
            raise ValueError("resolved cannot exceed cases.")
        cfr_corr = deaths / resolved if resolved > 0 else 0.0
        ci_lo_c = _st.beta.ppf(alpha / 2, deaths, resolved - deaths + 1) if deaths > 0 else 0.0
        ci_hi_c = _st.beta.ppf(1 - alpha / 2, deaths + 1, resolved - deaths) if deaths < resolved else 1.0
        result["cfr_corrected"] = float(cfr_corr)
        result["ci_lower_corrected"] = float(ci_lo_c)
        result["ci_upper_corrected"] = float(ci_hi_c)
        result["resolved"] = resolved

    return result


cfrir = case_fatality_rate


def cheatsheet() -> str:
    return "case_fatality_rate({}) -> CFR with interval-censoring correction."
