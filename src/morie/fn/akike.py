# morie.fn — function file (hadesllm/morie)
"""Akaike Information Criterion with R-style verbose result."""


def akike(loglik: float, k: int):
    """Akaike Information Criterion: AIC = 2k - 2 log L."""
    from ._richresult import RichResult
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")
    aic = 2.0 * k - 2.0 * loglik
    return RichResult(
        title="Akaike Information Criterion (AIC)",
        summary_lines=[
            ("AIC", aic),
            ("Log-likelihood", loglik),
            ("Free parameters k", k),
            ("Penalty (2k)", 2 * k),
        ],
        interpretation=("Lower is better. Compare to BIC (`bayic`) for the "
                        "Bayesian-flavored alternative; BIC penalises complexity "
                        "more strongly when n>=8."),
        payload={"value": aic, "statistic": aic, "loglik": loglik, "k": k},
    )
