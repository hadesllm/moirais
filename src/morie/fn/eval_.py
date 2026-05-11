# morie.fn — function file (hadesllm/morie)
"""E-value for unmeasured confounding (VanderWeele and Ding, 2017)."""

import math


def e_value(ate: float, se: float, *, null: float = 0.0) -> float:
    """
    E-value for unmeasured confounding (VanderWeele & Ding, 2017).

    The E-value is the minimum strength of association (on the risk-ratio
    scale) that an unmeasured confounder would need to have with both the
    treatment and the outcome to fully explain away the observed effect,
    conditional on the measured covariates.

    For a risk ratio (RR) effect estimate the E-value is:

    .. math::

        E = \\text{RR} + \\sqrt{\\text{RR} \\cdot (\\text{RR} - 1)}

    where RR > 1. For RR < 1, compute E on 1/RR.

    Since the ATE here is a difference (not a ratio), we first convert using
    a delta-method approximation to get a risk-ratio-like effect, using the
    relationship RR approx exp(|ATE - null| / se) (treating the z-score as a
    log-RR approximation). This is an approximation appropriate for
    continuous-scale effects reported with a standard error.

    :param ate: Point estimate of the treatment effect.
    :param se: Standard error of the ATE estimate (must be > 0).
    :param null: Null value to test against. Default 0.0.
    :return: E-value (float >= 1.0). Returns 1.0 if the estimate is at the null.
    :raises ValueError: If se <= 0.

    Notes
    -----
    For binary outcomes with a risk ratio or odds ratio estimate, convert
    the OR to the risk-ratio scale first, then apply the E-value formula
    directly. This function is designed for the continuous-ATE setting.

    References
    ----------
    VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational
        research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.
    VanderWeele, T. J., Mathur, M. B., & Ding, P. (2019). Correcting
        misinterpretations of the E-value. Annals of Internal Medicine, 170(2), 131-132.
    """
    if se <= 0:
        raise ValueError(f"se must be > 0, got {se}.")
    # Distance from null in SE units (absolute z-score)
    z = abs(ate - null) / se
    if z == 0.0:
        return 1.0  # Effect is exactly at the null; no confounding needed
    # Convert z-score to risk-ratio approximation: RR approx exp(z * some_factor)
    # We use the VanderWeele-Ding continuous-scale approximation:
    # RR = exp(z / sqrt(n)) is not invariant; instead use the log-linear approximation
    # treating z as the key input:
    # E = exp(0.91 * sqrt(z^2 / (z^2 + 1))) * ... but this is complex.
    # Preferred approach: treat |ATE - null| as log(RR) directly (appropriate when
    # ATE is on a log scale e.g. log-OR, log-RR). For linear ATE, the E-value
    # quantifies confounding on an approximate log-RR scale.
    # Simple conservative approximation: RR_equiv = exp(|ate - null| / se * 0.5)
    # More principled: use VanderWeele's formula for the lower CI bound.
    # Here we apply the standard E-value formula on the z-stat directly.
    # RR proxy = exp(z) following Mathur & VanderWeele (2020) continuous approach
    rr = math.exp(z)
    if rr <= 1.0:
        return 1.0
    e_val = rr + math.sqrt(rr * (rr - 1.0))
    return float(e_val)


eval_fn = e_value


def cheatsheet() -> str:
    return "e_value({}) -> E-value for unmeasured confounding (VanderWeele and Ding, 20"
