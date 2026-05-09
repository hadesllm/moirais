# moirais.fn — function file (hadesllm/moirais)
"""McFadden pseudo-R^2 with R-style verbose result."""


def mcfadr(ll_full: float, ll_null: float):
    """McFadden pseudo-R^2 for (logistic) regression model fit.

    R^2 = 1 - LL_full / LL_null

    Returns RichResult; float(result) yields scalar R^2.
    Benchmarks: 0.2-0.4 indicates excellent fit (NOT same scale as OLS R^2).

    References
    ----------
    McFadden (1974); Weisburd et al. (2022) ch.4.
    """
    from ._richresult import RichResult
    if ll_null == 0:
        raise ValueError("null log-likelihood is zero - R^2 undefined.")
    r2 = float(1.0 - (ll_full / ll_null))
    if r2 < 0.1: bench = "weak"
    elif r2 < 0.2: bench = "fair"
    elif r2 < 0.4: bench = "good"
    else: bench = "excellent"
    warnings = []
    if ll_full < ll_null:
        warnings.append("full LL < null LL: full model fits WORSE than intercept-only "
                        "(R^2 < 0). Check parameterization.")
    return RichResult(
        title="McFadden pseudo-R^2",
        summary_lines=[
            ("R^2 (McFadden)", r2), ("Benchmark", bench),
            ("LL(full)", ll_full), ("LL(null)", ll_null),
            ("LL ratio", ll_full / ll_null if ll_null != 0 else float("nan")),
        ],
        warnings=warnings,
        interpretation=(f"McFadden R^2={r2:.3f} -> {bench} fit per McFadden's "
                        ".2-.4 = excellent benchmark (NOT comparable to OLS R^2)."),
        payload={"value": r2, "statistic": r2, "benchmark": bench},
    )
