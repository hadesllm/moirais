# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Corrected AIC. 'The greatest teacher, failure is.'"""

from __future__ import annotations

from ._containers import DescriptiveResult


def corrected_aic(
    loglik: float,
    n: int,
    k: int,
) -> DescriptiveResult:
    """Corrected Akaike Information Criterion (AICc; Hurvich & Tsai, 1989).

    AICc = -2 * loglik + 2k + 2k(k+1) / (n - k - 1)

    Adds a small-sample correction to AIC, converges to AIC as n -> inf.

    :param loglik: Maximised log-likelihood.
    :param n: Number of observations.
    :param k: Number of estimated parameters.
    :return: DescriptiveResult with AICc value.
    """
    aic = -2.0 * loglik + 2.0 * k
    if n - k - 1 > 0:
        correction = 2.0 * k * (k + 1) / (n - k - 1)
    else:
        correction = float("inf")
    aicc_val = aic + correction
    return DescriptiveResult(
        name="aicc",
        value=float(aicc_val),
        extra={"aic": float(aic), "correction": float(correction), "loglik": loglik, "n": n, "k": k},
    )


aicc = corrected_aic


def cheatsheet() -> str:
    return "corrected_aic({}) -> Corrected AIC. 'The greatest teacher, failure is.'"
