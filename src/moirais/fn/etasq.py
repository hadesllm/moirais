# moirais.fn — function file (hadesllm/moirais)
"""Eta-squared (one-way ANOVA effect size)."""

def etasq(ss_between: float, ss_total: float) -> float:
    """Eta-squared: SS_between / SS_total.

    Proportion of total variance explained by group membership.
    """
    if ss_total <= 0:
        raise ValueError("ss_total must be positive.")
    return ss_between / ss_total
