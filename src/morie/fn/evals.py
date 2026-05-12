# morie.fn — function file (hadesllm/morie)
"""E-value for unmeasured confounding."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def e_value(
    point_estimate: float,
    ci_lower: float | None = None,
) -> ESRes:
    r"""
    Compute the E-value for sensitivity to unmeasured confounding.

    The E-value is the minimum strength of association that an
    unmeasured confounder would need to have with both treatment and
    outcome to explain away the observed effect.

    .. math::

        E = RR + \\sqrt{RR \\times (RR - 1)}

    Parameters
    ----------
    point_estimate : float
        Observed risk ratio (or OR/HR as approximation). Must be > 0.
    ci_lower : float, optional
        Lower CI bound. E-value for CI is computed if provided.

    Returns
    -------
    ESRes
        estimate = E-value for point estimate.

    References
    ----------
    VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in
    observational research: introducing the E-value. *Ann Intern Med*,
    167(4), 268-274.
    """
    if point_estimate <= 0:
        raise ValueError("point_estimate must be positive.")

    rr = point_estimate if point_estimate >= 1 else 1 / point_estimate
    e_pt = rr + np.sqrt(rr * (rr - 1))

    e_ci = None
    if ci_lower is not None and ci_lower > 0:
        rr_ci = ci_lower if ci_lower >= 1 else 1 / ci_lower
        if rr_ci <= 1:
            e_ci = 1.0
        else:
            e_ci = float(rr_ci + np.sqrt(rr_ci * (rr_ci - 1)))

    return ESRes(
        measure="e_value",
        estimate=float(e_pt),
        extra={
            "e_value_point": float(e_pt),
            "e_value_ci": e_ci,
            "point_estimate": point_estimate,
        },
    )


evals = e_value


def cheatsheet() -> str:
    return "e_value({}) -> E-value for unmeasured confounding."
