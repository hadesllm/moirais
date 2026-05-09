# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""AIC from entropy."""


from ._containers import ESRes


def aic_entropy(log_likelihood: float, k: int, n: int = 0, **kwargs) -> ESRes:
    """
    Compute AIC and AICc from log-likelihood.

    .. math::

        AIC = 2k - 2\\ln(L)

    .. math::

        AICc = AIC + \\frac{2k(k+1)}{n - k - 1}

    :param log_likelihood: Log-likelihood of the model.
    :param k: Number of estimated parameters.
    :param n: Sample size (for AICc correction). If 0, AICc not computed.
    :return: ESRes with AIC (and AICc if n > 0).

    References
    ----------
    Akaike H (1974). A new look at the statistical model identification.
    IEEE Transactions on Automatic Control, 19(6), 716-723.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    aic = 2.0 * k - 2.0 * log_likelihood
    extra: dict = {"k": k, "log_likelihood": log_likelihood}
    if n > k + 1:
        aicc = aic + (2.0 * k * (k + 1)) / (n - k - 1)
        extra["AICc"] = aicc
    return ESRes(
        measure="aic_entropy",
        estimate=float(aic),
        n=n if n > 0 else None,
        extra=extra,
    )


aient = aic_entropy


def cheatsheet() -> str:
    return "aic_entropy(log_likelihood, k, n) -> AIC from entropy."
