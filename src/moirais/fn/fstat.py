# moirais.fn — function file (hadesllm/moirais)
"""F-statistic for nested-model comparison (atomic, scalar return)."""


def fstat(ssr_val: float, sse_val: float, k: int, n: int) -> float:
    """F-statistic: (SSR / k) / (SSE / (n - k - 1)).

    Atomic primitive — returns scalar. For full F-test output use a
    full ANOVA or call lrtst().
    """
    if sse_val <= 0:
        raise ValueError(f"SSE must be positive, got {sse_val}.")
    if k < 1 or n - k - 1 < 1:
        raise ValueError(f"invalid k={k} or n={n}.")
    return (ssr_val / k) / (sse_val / (n - k - 1))
