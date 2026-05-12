# morie.fn -- function file (hadesllm/morie)
"""Omega-squared (less biased than η²)."""

def omeg2(ss_between: float, df_between: int,
           ms_within: float, ss_total: float) -> float:
    """Omega-squared: less biased than η² for small samples.

    ω² = (SS_b − df_b × MS_w) / (SS_t + MS_w)
    """
    if ss_total + ms_within == 0:
        raise ValueError("denominator is zero.")
    return (ss_between - df_between * ms_within) / (ss_total + ms_within)
