# morie.fn — function file (hadesllm/morie)
"""t-statistic for one coefficient (atomic, scalar return)."""


def tstat(estimate: float, std_error: float, null_value: float = 0.0) -> float:
    """t-statistic: (estimate - H0) / SE.

    Atomic primitive — returns scalar. For full hypothesis-test output
    use `wald` (with test='z') or wrap in welcht/paired.
    """
    if std_error <= 0:
        raise ValueError(f"std_error must be positive, got {std_error}.")
    return (estimate - null_value) / std_error
