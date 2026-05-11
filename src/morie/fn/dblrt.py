# morie.fn — function file (hadesllm/morie)
"""Population doubling time."""

import numpy as np

from ._containers import ESRes


def population_doubling(
    growth_rate: float,
) -> ESRes:
    """Compute population doubling time.

    .. math::

        t_d = \\frac{\\ln 2}{r}

    Parameters
    ----------
    growth_rate : float
        Annual growth rate (proportion, e.g. 0.02 for 2%).

    Returns
    -------
    ESRes
    """
    if growth_rate <= 0:
        return ESRes(
            measure="doubling_time",
            estimate=float("inf"),
            extra={"growth_rate": float(growth_rate), "note": "non-positive growth"},
        )

    td = np.log(2) / growth_rate

    return ESRes(
        measure="doubling_time",
        estimate=float(td),
        extra={"growth_rate": float(growth_rate), "rule_of_70": float(70 / (growth_rate * 100))},
    )


dblrt = population_doubling


def cheatsheet() -> str:
    return "population_doubling({}) -> Population doubling time."
