# morie.fn — function file (hadesllm/morie)
"""Fuzzy entropy."""

import numpy as np

from ._containers import ESRes


def fuzzy_entropy(x, m: int = 2, r: float | None = None, n_exp: float = 2.0, **kwargs) -> ESRes:
    """
    Compute fuzzy entropy (FuzzyEn) of a time series.

    Uses fuzzy membership functions (exponential) instead of
    Heaviside step functions used in SampEn.

    :param x: 1-D array-like time series.
    :param m: Embedding dimension (default 2).
    :param r: Similarity tolerance (default 0.2 * std(x)).
    :param n_exp: Fuzzy exponent (default 2).
    :return: ESRes with fuzzy entropy.

    References
    ----------
    Chen W et al. (2007). Characterization of surface EMG signal based
    on fuzzy entropy. IEEE Trans Neural Syst Rehabil Eng, 15(2), 266-272.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < m + 2:
        raise ValueError(f"Need at least {m + 2} observations.")
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    if r <= 0:
        raise ValueError("Tolerance r must be positive.")

    def _phi_fuzzy(dim):
        templates = []
        for i in range(n - dim):
            seg = x[i:i + dim]
            templates.append(seg - np.mean(seg))
        templates = np.array(templates)
        nt = len(templates)
        total = 0.0
        for i in range(nt):
            for j in range(i + 1, nt):
                d = np.max(np.abs(templates[i] - templates[j]))
                total += np.exp(-((d / r) ** n_exp))
        denom = nt * (nt - 1) / 2
        return total / denom if denom > 0 else 0.0

    phi_m = _phi_fuzzy(m)
    phi_m1 = _phi_fuzzy(m + 1)

    if phi_m <= 0 or phi_m1 <= 0:
        fe = float("inf")
    else:
        fe = -np.log(phi_m1 / phi_m)

    return ESRes(
        measure="fuzzy_entropy",
        estimate=float(fe),
        n=n,
        extra={"m": m, "r": r, "n_exp": n_exp, "phi_m": phi_m, "phi_m1": phi_m1},
    )


fuznt = fuzzy_entropy


def cheatsheet() -> str:
    return "fuzzy_entropy(x, m=2) -> Fuzzy entropy."
