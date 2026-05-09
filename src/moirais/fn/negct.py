# moirais.fn — function file (hadesllm/moirais)
"""Negative control outcome/exposure test."""

from __future__ import annotations

from ._containers import TestResult


def negative_control(
    estimate: float,
    negative_control_estimate: float,
    *,
    se_estimate: float = 0.1,
    se_negative: float = 0.1,
) -> TestResult:
    """
    Test for residual confounding using a negative control.

    If the negative control estimate is significantly different from
    zero, residual confounding is likely present.

    Parameters
    ----------
    estimate : float
        Primary causal estimate.
    negative_control_estimate : float
        Estimate for the negative control outcome/exposure.
    se_estimate : float
        SE of primary estimate.
    se_negative : float
        SE of negative control estimate.

    Returns
    -------
    TestResult

    References
    ----------
    Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010).
    Negative controls: a tool for detecting confounding and bias in
    observational studies. *Epidemiology*, 21(3), 383-388.
    """
    z_neg = negative_control_estimate / se_negative if se_negative > 0 else 0
    from scipy.stats import norm

    p_neg = float(2 * norm.sf(abs(z_neg)))

    return TestResult(
        test_name="negative_control",
        statistic=float(z_neg),
        p_value=p_neg,
        df=None,
        method="Negative control Z-test",
        extra={
            "primary_estimate": estimate,
            "negative_control_estimate": negative_control_estimate,
            "confounding_suspected": p_neg < 0.05,
        },
    )


negct = negative_control


def cheatsheet() -> str:
    return "negative_control({}) -> Negative control outcome/exposure test."
