# morie.fn — function file (hadesllm/morie)
"""QQ-plot data for GWAS p-values."""

import numpy as np

from ._containers import DescriptiveResult


def qq_plot_data(p_values: np.ndarray) -> DescriptiveResult:
    """
    Prepare quantile-quantile plot data for GWAS p-values.

    Computes observed vs expected -log10(p) values for assessing
    genomic inflation.

    :param p_values: Array of p-values from association tests.
    :return: DescriptiveResult with expected and observed in extra.
    :raises ValueError: If any p-value is outside (0, 1].

    References
    ----------
    Devlin B, Roeder K (1999). Genomic control for association studies.
    Biometrics, 55(4), 997-1004.
    """
    pv = np.asarray(p_values, dtype=np.float64).ravel()
    if np.any(pv <= 0) or np.any(pv > 1):
        raise ValueError("p-values must be in (0, 1].")
    n = len(pv)
    observed = -np.log10(np.sort(pv))
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    lambda_gc = float(np.median(observed) / np.median(expected)) if np.median(expected) > 0 else 1.0
    return DescriptiveResult(
        name="qq_plot_data",
        value=lambda_gc,
        extra={"observed": observed, "expected": expected, "n": n, "lambda_gc": lambda_gc},
    )


qq_ = qq_plot_data


def cheatsheet() -> str:
    return "qq_plot_data({}) -> QQ-plot data for GWAS p-values."
