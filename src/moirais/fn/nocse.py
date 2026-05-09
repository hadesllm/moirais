# moirais.fn — function file (hadesllm/moirais)
"""NOMINATE confidence intervals from bootstrap SEs."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nominate_confidence_interval(boot_se, alpha=0.05) -> DescriptiveResult:
    """Compute CIs from bootstrap standard errors.

    .. epigraph:: "Better call Saul!" -- Saul, Better Call Saul
    """
    import numpy as np
    from scipy.stats import norm

    se = np.asarray(boot_se, dtype=float)
    z = norm.ppf(1 - alpha / 2)
    ci_half = z * se
    return DescriptiveResult(
        name="nominate_confidence_interval",
        value=float(z),
        extra={
            "z_critical": float(z),
            "alpha": alpha,
            "ci_half_widths": ci_half,
            "mean_se": float(np.mean(se)),
            "n_params": len(se),
        },
    )


nocse = nominate_confidence_interval


def cheatsheet() -> str:
    return "nominate_confidence_interval({}) -> NOMINATE confidence intervals from bootstrap SEs."
