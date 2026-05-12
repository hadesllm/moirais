# morie.fn -- function file (hadesllm/morie)
"""Mantel-Haenszel pooled odds ratio."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def mantel_haenszel_or(
    tables: list[tuple[int, int, int, int]],
    confidence: float = 0.95,
) -> ESRes:
    r"""Mantel-Haenszel pooled odds ratio across K 2x2 tables.

    .. math::

        OR_{MH} = \\frac{\\sum_k a_k d_k / n_k}{\\sum_k b_k c_k / n_k}

    Parameters
    ----------
    tables : list of (a, b, c, d) tuples
        Each tuple is a 2x2 table stratum.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes

    References
    ----------
    Mantel, N. & Haenszel, W. (1959). Statistical aspects of the
    analysis of data from retrospective studies of disease.
    Journal of the National Cancer Institute, 22(4), 719-748.

    Robins, J. et al. (1986). Estimators of the Mantel-Haenszel
    variance consistent in both sparse data and large-strata
    limiting models. Biometrics, 42(2), 311-323.
    """
    if not tables:
        raise ValueError("At least one table required")

    num = 0.0
    den = 0.0
    P = 0.0
    Q = 0.0
    R = 0.0
    S = 0.0

    for a, b, c, d in tables:
        n_k = a + b + c + d
        if n_k == 0:
            continue
        num += a * d / n_k
        den += b * c / n_k
        P += (a + d) * a * d / n_k**2
        Q += ((a + d) * b * c + (b + c) * a * d) / n_k**2
        R += (b + c) * b * c / n_k**2
        S += a * d / n_k

    if den == 0:
        raise ValueError("Denominator is zero")

    or_mh = num / den
    T = b * c / (a + b + c + d) if len(tables) == 1 else den

    var_ln = P / (2 * S**2) + Q / (2 * S * T) + R / (2 * T**2)
    se_ln = np.sqrt(var_ln) if var_ln > 0 else 0.0

    z = stats.norm.ppf((1 + confidence) / 2)
    ln_or = np.log(or_mh)

    return ESRes(
        measure="OR_MH",
        estimate=float(or_mh),
        se=float(se_ln),
        ci_lower=float(np.exp(ln_or - z * se_ln)),
        ci_upper=float(np.exp(ln_or + z * se_ln)),
        n=sum(a + b + c + d for a, b, c, d in tables),
        extra={"n_strata": len(tables)},
    )


mhors = mantel_haenszel_or


def cheatsheet() -> str:
    return "mantel_haenszel_or({}) -> Mantel-Haenszel pooled odds ratio."
