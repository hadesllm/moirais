# morie.fn — function file (hadesllm/morie)
"""Bayesian Information Criterion with R-style verbose result."""

import math


def bayic(loglik: float, k: int, n: int):
    """BIC = k log(n) - 2 log L."""
    from ._richresult import RichResult
    if k < 0 or n < 1:
        raise ValueError(f"k must be >=0 and n >=1; got k={k}, n={n}.")
    bic = k * math.log(n) - 2.0 * loglik
    aic = 2.0 * k - 2.0 * loglik
    return RichResult(
        title="Bayesian (Schwarz) Information Criterion (BIC)",
        summary_lines=[
            ("BIC", bic),
            ("AIC (for context)", aic),
            ("BIC - AIC penalty diff", bic - aic),
            ("Log-likelihood", loglik),
            ("Parameters k", k), ("n observations", n),
            ("Penalty (k log n)", k * math.log(n)),
        ],
        interpretation=("Lower is better. BIC penalises complexity more than "
                        "AIC for n>=8 (penalty grows with sample size). "
                        "Approximation to log Bayes factor (see `bayesf`)."),
        payload={"value": bic, "statistic": bic, "aic": aic,
                 "loglik": loglik, "k": k, "n": n},
    )
