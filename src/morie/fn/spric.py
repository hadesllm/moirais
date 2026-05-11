"""Species richness estimation (Chao1)."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Life is really simple, but we insist on making it complicated. — Confucius"


def species_richness(samples, method: str = "chao1", **kwargs) -> DescriptiveResult:
    """
    Estimate species richness using the Chao1 or ACE estimator.

    The Chao1 lower-bound estimator:

    .. math::

        S_{\\text{Chao1}} = S_{\\text{obs}} + \\frac{f_1 (f_1 - 1)}{2(f_2 + 1)}

    where :math:`f_1` = singletons and :math:`f_2` = doubletons.

    :param samples: Array-like of species counts (non-negative integers).
    :param method: ``"chao1"`` (default) or ``"observed"``.
    :return: DescriptiveResult with estimated richness.
    :raises ValueError: If method is unknown or counts are negative.

    References
    ----------
    Chao, A. (1984). Nonparametric estimation of the number of classes in
    a population. *Scandinavian Journal of Statistics*, 11(4), 265-270.
    """
    samples = np.asarray(samples, dtype=np.float64).ravel()
    if np.any(samples < 0):
        raise ValueError("Counts must be non-negative.")

    counts = samples[samples > 0]
    s_obs = int(len(counts))

    if method == "observed":
        s_est = float(s_obs)
    elif method == "chao1":
        f1 = float(np.sum(counts == 1))
        f2 = float(np.sum(counts == 2))
        if f2 > 0:
            s_est = s_obs + (f1 * (f1 - 1)) / (2.0 * (f2 + 1))
        else:
            s_est = s_obs + (f1 * (f1 - 1)) / 2.0 if f1 > 1 else float(s_obs)
    else:
        raise ValueError(f"method must be 'chao1' or 'observed', got '{method}'.")

    f1 = float(np.sum(counts == 1))
    f2 = float(np.sum(counts == 2))

    return DescriptiveResult(
        name="species_richness",
        value=s_est,
        extra={
            "S_observed": s_obs,
            "S_estimated": s_est,
            "singletons_f1": f1,
            "doubletons_f2": f2,
            "method": method,
            "total_individuals": float(np.sum(samples)),
        },
    )


spric = species_richness


def cheatsheet() -> str:
    return "species_richness({}) -> Species richness estimation (Chao1)."
