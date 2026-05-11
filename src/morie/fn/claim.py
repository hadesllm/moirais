# morie.fn — function file (hadesllm/morie)
"""Poisson claim frequency model."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def claim_frequency(
    n_claims: np.ndarray,
    exposure: np.ndarray,
    confidence: float = 0.95,
) -> DescriptiveResult:
    """Estimate claim frequency under a Poisson model.

    The MLE of the Poisson rate parameter is:

    .. math::

        \\hat{\\lambda} = \\frac{\\sum_i n_i}{\\sum_i e_i}

    with an exact confidence interval based on the chi-squared
    distribution.

    Parameters
    ----------
    n_claims : array-like
        Number of claims per policy/unit.
    exposure : array-like
        Exposure (e.g. policy-years) per unit.
    confidence : float, default 0.95
        Confidence level for the rate CI.

    Returns
    -------
    DescriptiveResult
        ``value`` is the estimated claim rate.  ``extra`` has
        ``total_claims``, ``total_exposure``, ``ci_lower``,
        ``ci_upper``, ``dispersion`` (variance/mean ratio).

    References
    ----------
    Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2012).
    *Loss Models: From Data to Decisions* (4th ed.). Wiley.
    """
    nc = np.asarray(n_claims, dtype=np.float64).ravel()
    ex = np.asarray(exposure, dtype=np.float64).ravel()
    if len(nc) != len(ex):
        raise ValueError("n_claims and exposure must have same length.")
    if np.any(ex <= 0):
        raise ValueError("All exposure values must be positive.")

    total_claims = float(np.sum(nc))
    total_exposure = float(np.sum(ex))
    rate = total_claims / total_exposure

    alpha = 1 - confidence
    ci_lo = _st.chi2.ppf(alpha / 2, 2 * total_claims) / (2 * total_exposure)
    ci_hi = _st.chi2.ppf(1 - alpha / 2, 2 * (total_claims + 1)) / (2 * total_exposure)

    per_unit_rate = nc / ex
    dispersion = float(np.var(per_unit_rate) / np.mean(per_unit_rate)) if np.mean(per_unit_rate) > 0 else 0.0

    return DescriptiveResult(
        name="ClaimFrequency",
        value=float(rate),
        extra={
            "total_claims": total_claims,
            "total_exposure": total_exposure,
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "dispersion": dispersion,
            "n_policies": len(nc),
        },
    )


claim = claim_frequency


def cheatsheet() -> str:
    return "claim_frequency({}) -> Poisson claim frequency model."
