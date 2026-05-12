# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bonferroni correction for multiple comparisons."""

import numpy as np

from ._containers import DescriptiveResult


def bonferroni_correction(
    p_values: np.ndarray | list,
    *,
    alpha: float = 0.05,
) -> DescriptiveResult:
    """Bonferroni correction: adjusted p_i = min(p_i * m, 1.0).

    The simplest and most conservative multiple-testing correction.
    Controls the family-wise error rate (FWER).

    Parameters
    ----------
    p_values : array-like
        Raw p-values from m hypothesis tests.
    alpha : float, default 0.05
        Significance threshold applied to adjusted p-values.

    Returns
    -------
    DescriptiveResult
        name="Bonferroni", value=number of significant tests,
        extra contains 'adjusted' (array of adjusted p-values)
        and 'significant' (boolean mask).

    Raises
    ------
    ValueError
        If p_values is empty or any value outside [0, 1].

    References
    ----------
    Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo
        delle probabilita. Pubblicazioni del R Istituto Superiore di
        Scienze Economiche e Commerciali di Firenze, 8, 3-62.
    """
    pv = np.asarray(p_values, dtype=float)
    if len(pv) == 0:
        raise ValueError("p_values must not be empty.")
    if np.any(pv < 0) or np.any(pv > 1):
        raise ValueError("All p-values must be in [0, 1].")

    m = len(pv)
    adjusted = np.minimum(pv * m, 1.0)
    sig = adjusted < alpha

    return DescriptiveResult(
        name="Bonferroni",
        value=int(np.sum(sig)),
        extra={
            "adjusted": adjusted.tolist(),
            "significant": sig.tolist(),
            "m": m,
            "alpha": alpha,
        },
    )


bonf = bonferroni_correction


def cheatsheet() -> str:
    return "bonferroni_correction({}) -> Bonferroni correction for multiple comparisons."
