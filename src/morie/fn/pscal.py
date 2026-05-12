# morie.fn -- function file (hadesllm/morie)
"""An unexamined life is not worth living. -- Socrates"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ps_calibrate(
    ps: np.ndarray,
    treatment: np.ndarray,
) -> DescriptiveResult:
    r"""
    Calibrate propensity scores using Platt scaling.

    Fits :math:`P(T=1|ps) = \\text{logistic}(a \\cdot ps + b)` to
    improve the calibration of estimated propensity scores.

    :param ps: Estimated propensity scores (n,).
    :param treatment: Binary treatment indicator (n,), 0/1.
    :return: DescriptiveResult with Brier score as value.
    :raises ValueError: If inputs invalid.

    References
    ----------
    Platt, J. (2000). Probabilistic outputs for support vector machines
    and comparisons to regularized likelihood methods. In Advances in
    Large Margin Classifiers, MIT Press, 61--74.
    """
    ps = np.asarray(ps, dtype=float)
    treatment = np.asarray(treatment, dtype=float).ravel()
    if ps.shape != treatment.shape or ps.size < 10:
        raise ValueError("ps and treatment must match in shape with n >= 10.")

    def _logistic(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    from scipy.optimize import minimize

    def neg_loglik(params):
        a, b = params
        p = _logistic(a * ps + b)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.sum(treatment * np.log(p) + (1 - treatment) * np.log(1 - p))

    res = minimize(neg_loglik, [1.0, 0.0], method="Nelder-Mead")
    a_hat, b_hat = res.x
    ps_calibrated = _logistic(a_hat * ps + b_hat)

    brier_before = float(np.mean((ps - treatment) ** 2))
    brier_after = float(np.mean((ps_calibrated - treatment) ** 2))

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    cal_table = []
    for i in range(n_bins):
        mask = (ps_calibrated >= bin_edges[i]) & (ps_calibrated < bin_edges[i + 1])
        if np.sum(mask) > 0:
            cal_table.append(
                {
                    "bin": f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}",
                    "n": int(np.sum(mask)),
                    "mean_predicted": float(np.round(np.mean(ps_calibrated[mask]), 4)),
                    "mean_observed": float(np.round(np.mean(treatment[mask]), 4)),
                }
            )

    return DescriptiveResult(
        name="PS Calibration",
        value=float(np.round(brier_after, 6)),
        extra={
            "brier_before": float(np.round(brier_before, 6)),
            "brier_after": float(np.round(brier_after, 6)),
            "platt_a": float(np.round(a_hat, 4)),
            "platt_b": float(np.round(b_hat, 4)),
            "ps_calibrated": ps_calibrated,
            "calibration_table": cal_table,
            "n": len(ps),
        },
    )


pscal = ps_calibrate


def cheatsheet() -> str:
    return "An unexamined life is not worth living. -- Socrates"
