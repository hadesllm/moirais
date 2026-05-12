# morie.fn -- function file (hadesllm/morie)
"""Hedges' g (bias-corrected Cohen's d) with R-style verbose result."""


def hedgeg(cohens_d: float, n1: int, n2: int):
    """Hedges' g: small-sample correction of Cohen's d."""
    from ._richresult import RichResult
    df = n1 + n2 - 2
    if df < 1:
        raise ValueError(f"need n1+n2 >= 3; got n1={n1}, n2={n2}.")
    J = 1 - 3.0 / (4 * (n1 + n2) - 9)
    g = cohens_d * J
    abs_g = abs(g)
    if abs_g < 0.2: bench = "negligible"
    elif abs_g < 0.5: bench = "small"
    elif abs_g < 0.8: bench = "medium"
    else: bench = "large"
    return RichResult(
        title="Hedges' g (bias-corrected Cohen's d)",
        summary_lines=[
            ("Hedges' g", g),
            ("Original Cohen's d", cohens_d),
            ("Correction factor J", J),
            ("|g| benchmark", bench),
            ("n_1", n1), ("n_2", n2), ("df", df),
        ],
        interpretation=(f"g = d × J = {cohens_d:.3f} × {J:.4f} = {g:.3f}. "
                        "g is preferred over d for small samples (n_total < 50)."),
        payload={"value": g, "statistic": g, "J": J, "benchmark": bench},
    )
