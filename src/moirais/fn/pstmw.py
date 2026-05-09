# moirais.fn — function file (hadesllm/moirais)
"""Trimmed IPW weights. 'Luminous beings are we, not this crude matter.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ps_trimmed_weights(
    ps: np.ndarray,
    treatment: np.ndarray,
    trim_pct: float = 0.01,
) -> DescriptiveResult:
    """
    Compute trimmed inverse probability weights (IPW).

    Trims extreme propensity scores to [trim_pct, 1 - trim_pct]
    before computing ATE weights.

    .. math::

        w_i^{ATE} = \\frac{T_i}{e(X_i)} + \\frac{1 - T_i}{1 - e(X_i)}

    :param ps: Estimated propensity scores (n,).
    :param treatment: Binary treatment indicator (n,), 0/1.
    :param trim_pct: Trim percentage at each tail. Default 0.01 (1%).
    :return: DescriptiveResult with trimmed weights.
    :raises ValueError: If inputs invalid.

    References
    ----------
    Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009).
    Dealing with limited overlap in estimation of average treatment
    effects. Biometrika, 96(1), 187--199. doi:10.1093/biomet/asn055
    """
    ps = np.asarray(ps, dtype=float)
    treatment = np.asarray(treatment, dtype=float).ravel()
    if ps.shape != treatment.shape:
        raise ValueError("ps and treatment must have the same shape.")
    if not 0.0 <= trim_pct < 0.5:
        raise ValueError(f"trim_pct must be in [0, 0.5), got {trim_pct}.")

    ps_trimmed = np.clip(ps, trim_pct, 1.0 - trim_pct)
    n_trimmed = int(np.sum((ps < trim_pct) | (ps > 1.0 - trim_pct)))

    weights = np.where(
        treatment == 1,
        1.0 / ps_trimmed,
        1.0 / (1.0 - ps_trimmed),
    )

    weights = weights / np.mean(weights)

    return DescriptiveResult(
        name="Trimmed IPW Weights",
        value=float(np.round(np.mean(weights), 4)),
        extra={
            "weights": weights,
            "n_trimmed": n_trimmed,
            "trim_pct": trim_pct,
            "ess_treated": float(
                np.round(np.sum(weights[treatment == 1]) ** 2 / np.sum(weights[treatment == 1] ** 2), 1)
            ),
            "ess_control": float(
                np.round(np.sum(weights[treatment == 0]) ** 2 / np.sum(weights[treatment == 0] ** 2), 1)
            ),
            "max_weight": float(np.round(np.max(weights), 4)),
            "n": len(ps),
        },
    )


pstmw = ps_trimmed_weights


def cheatsheet() -> str:
    return "ps_trimmed_weights({}) -> Trimmed IPW weights. 'Luminous beings are we, not this crude"
