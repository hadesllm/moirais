# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Asymptotic relative efficiency (ARE) for nonparametric tests."""


__all__ = ["areff"]


def areff(test1_stat, test2_stat, n, test1_name="Test1", test2_name="Test2"):
    r"""
    Asymptotic relative efficiency between two tests.

    ARE = (asymptotic variance of test2) / (asymptotic variance of test1)
    Indicates how many more observations test1 needs to match test2's power.
    """
    test1_stat = float(test1_stat)
    test2_stat = float(test2_stat)

    if test1_stat <= 0 or test2_stat <= 0:
        raise ValueError("Test statistics must be positive")

    # Simplified ARE based on variance ratios
    # In practice, would use theoretical asymptotic variances
    are = test2_stat / test1_stat

    return {
        "are": float(are),
        "interpretation": f"{test1_name} is {are:.2f}x as efficient as {test2_name}",
        "n_ratio": float(1 / are),
    }
