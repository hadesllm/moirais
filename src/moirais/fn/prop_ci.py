# moirais.fn — function file (hadesllm/moirais)
"""Confidence interval for a single proportion."""

import math

import scipy.stats as stats


def proportion_ci(
    successes: int,
    n: int,
    *,
    method: str = "wilson",
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Confidence interval for a single proportion.

    Three methods are supported:
    - ``"wilson"`` (default): Wilson score interval -- recommended for small n
      and proportions near 0 or 1 (Brown, Cai & DasGupta, 2001).
    - ``"clopper-pearson"``: Exact Clopper-Pearson interval -- conservative.
    - ``"agresti-coull"``: Agresti-Coull interval -- good coverage for moderate n.

    :param successes: Number of successes (0 <= successes <= n).
    :param n: Total number of trials (> 0).
    :param method: CI method. Default ``"wilson"``.
    :param alpha: Significance level (default 0.05 -> 95% CI).
    :return: Tuple ``(lower, upper)``.
    :raises ValueError: If successes < 0, n <= 0, or alpha not in (0, 1).

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
        statistical inference. JASA, 22, 209-212.
    Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial
        limits illustrated in the case of the binomial. Biometrika, 26, 404-413.
    Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a
        binomial proportion. Statistical Science, 16(2), 101-133.
    """
    if successes < 0:
        raise ValueError("successes must be >= 0.")
    if n <= 0:
        raise ValueError("n must be > 0.")
    if successes > n:
        raise ValueError("successes cannot exceed n.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    p_hat = successes / n
    z = float(stats.norm.ppf(1 - alpha / 2))

    if method == "wilson":
        denom = 1 + z**2 / n
        centre = (p_hat + z**2 / (2 * n)) / denom
        half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
        return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))
    elif method == "clopper-pearson":
        lower = float(stats.beta.ppf(alpha / 2, successes, n - successes + 1)) if successes > 0 else 0.0
        upper = float(stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)) if successes < n else 1.0
        return lower, upper
    elif method == "agresti-coull":
        n_tilde = n + z**2
        p_tilde = (successes + z**2 / 2) / n_tilde
        half_width = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        return float(max(0.0, p_tilde - half_width)), float(min(1.0, p_tilde + half_width))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'wilson', 'clopper-pearson', or 'agresti-coull'.")


prop_ci = proportion_ci


def cheatsheet() -> str:
    return "proportion_ci({}) -> Confidence interval for a single proportion."
