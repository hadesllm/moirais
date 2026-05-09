"""Years of potential life lost (YPLL)."""

import numpy as np

from ._containers import ESRes


def years_potential_life_lost(
    ages_at_death: np.ndarray,
    cutoff: float = 75.0,
) -> ESRes:
    """Years of potential life lost before a reference age.

    .. math::

        YPLL = \\sum_{i} \\max(0,\\; \\text{cutoff} - \\text{age}_i)

    Parameters
    ----------
    ages_at_death : array-like
        Ages at death for each decedent.
    cutoff : float, default 75
        Reference age (deaths at or above this age contribute 0).

    Returns
    -------
    ESRes

    References
    ----------
    Gardner, J. W. & Sanborn, J. S. (1990). Years of potential life lost
    (YPLL) -- what does it measure? Epidemiology, 1(4), 322-329.
    """
    ages = np.asarray(ages_at_death, dtype=float)
    ypll_per = np.maximum(0, cutoff - ages)
    total = float(np.sum(ypll_per))
    n_premature = int(np.sum(ypll_per > 0))

    return ESRes(
        measure="YPLL",
        estimate=total,
        n=len(ages),
        extra={
            "cutoff": cutoff,
            "n_premature": n_premature,
            "mean_ypll": float(total / len(ages)) if len(ages) > 0 else 0.0,
        },
    )


ypll = years_potential_life_lost


def cheatsheet() -> str:
    return "years_potential_life_lost({}) -> Years of potential life lost (YPLL)."
