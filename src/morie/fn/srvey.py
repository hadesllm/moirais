"""Survey prevalence estimate with design effect."""

import numpy as np

from ._containers import ESRes


def survey_prevalence(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    confidence: float = 0.95,
) -> ESRes:
    """Weighted survey prevalence estimate with design effect.

    Computes the weighted proportion and its standard error, adjusting
    for the design effect (deff = variance under weighting / variance
    under SRS).

    Parameters
    ----------
    values : array-like
        Binary (0/1) outcome values.
    weights : array-like or None
        Survey weights. If None, equal weights assumed.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes
        estimate = weighted proportion. extra contains deff, ess.

    References
    ----------
    Kish, L. (1965). Survey Sampling. Wiley.
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n == 0:
        raise ValueError("values must not be empty")

    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != n:
            raise ValueError("values and weights must have equal length")

    w_sum = np.sum(w)
    p_hat = np.sum(w * vals) / w_sum

    deff = n * np.sum(w**2) / w_sum**2
    ess = n / deff

    se_srs = np.sqrt(p_hat * (1 - p_hat) / n) if n > 0 else 0.0
    se = se_srs * np.sqrt(deff)

    from scipy.stats import norm

    z = norm.ppf((1 + confidence) / 2)

    return ESRes(
        measure="Survey prevalence",
        estimate=float(p_hat),
        ci_lower=float(max(0.0, p_hat - z * se)),
        ci_upper=float(min(1.0, p_hat + z * se)),
        se=float(se),
        n=n,
        extra={"deff": float(deff), "ess": float(ess)},
    )


srvey = survey_prevalence


def cheatsheet() -> str:
    return "survey_prevalence({}) -> Survey prevalence estimate with design effect."
