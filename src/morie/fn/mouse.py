# morie.fn -- function file (hadesllm/morie)
"""Monte Carlo simulation engine. 'To deny our own impulses is to deny the very thing that makes us human.' -- Mouse"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def monte_carlo(
    fn_expr: str = "x**2",
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    n_samples: int = 10000,
    seed: int | None = 42,
) -> DescriptiveResult:
    """Monte Carlo integration of f(x) over [lo, hi].

    Estimates the integral by sampling uniformly in [lo, hi] and computing
    the sample mean of f(x) * (hi - lo).

    Parameters
    ----------
    fn_expr : str
        A Python expression in variable ``x`` (e.g. ``"np.sin(x)"``).
        Only ``np``, ``x``, and basic math operators are available.
    lo, hi : float
        Integration bounds.
    n_samples : int
        Number of random samples.
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the estimated integral; ``extra`` has standard error
        and confidence interval.
    """
    if hi <= lo:
        raise ValueError("hi must be > lo")

    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, n_samples)

    safe_ns = {"np": np, "x": x, "__builtins__": {}}
    try:
        fx = eval(fn_expr, safe_ns)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Cannot evaluate expression '{fn_expr}': {exc}") from exc

    fx = np.asarray(fx, dtype=np.float64)
    width = hi - lo
    estimate = float(fx.mean() * width)
    se = float(fx.std(ddof=1) / np.sqrt(n_samples) * width)
    ci_lo = estimate - 1.96 * se
    ci_hi = estimate + 1.96 * se

    return DescriptiveResult(
        name="Monte Carlo Integration",
        value=estimate,
        extra={
            "se": se,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_samples": n_samples,
            "bounds": [lo, hi],
            "expression": fn_expr,
        },
    )


mouse = monte_carlo


def cheatsheet() -> str:
    return "monte_carlo({}) -> Monte Carlo simulation engine. 'To deny our own impulses is "
