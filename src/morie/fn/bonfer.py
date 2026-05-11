# morie.fn — function file (hadesllm/morie)
"""Bonferroni correction with R-style verbose result."""


def bonfer(alpha: float, m: int):
    """Bonferroni-corrected significance threshold: alpha' = alpha / m."""
    from ._richresult import RichResult
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    corrected = alpha / m
    return RichResult(
        title="Bonferroni multiple-testing correction",
        summary_lines=[
            ("Original alpha", alpha),
            ("Number of tests m", m),
            ("Corrected alpha (per test)", corrected),
            ("Familywise alpha (FWER)", alpha),
        ],
        warnings=[] if m <= 100 else
                 [f"m={m} comparisons; Bonferroni is conservative for large m. "
                  "Consider Holm (`holm`) or Benjamini-Hochberg (`bhfdr`) for "
                  "more power."],
        interpretation=(f"To control FWER at {alpha}, reject each test only if "
                        f"p < {corrected:.4g}."),
        payload={"value": corrected, "statistic": corrected,
                 "alpha": alpha, "m": m},
    )
